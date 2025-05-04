from typing_extensions import override
from openpi.models.pi0 import Pi0, Pi0Config, make_attn_mask
from openpi.shared import array_typing as at
import jax, jax.numpy as jnp
from openpi.models import model as _model

class IMLEConfig(Pi0Config):
    # imle specific
    num_noise_samples: int = 16 #k
    metric: str = "l2"   # "l2" or "mse"
    epsilon: float | tuple[float, float] = 0.03
    num_backprop: int = 1

    def create(self, rng):
        import flax.nnx as nnx
        return IMLEModel(self, rngs=nnx.Rngs(rng))

class IMLEModel(Pi0):
    """Pi0 backbone trained with rs imle"""

    def __init__(self, config: IMLEConfig, rngs):
        super().__init__(config, rngs)
        self.cfg: IMLEConfig = config



    def embed_suffix_imle(
        self,
        obs: _model.Observation,
        latent_actions: at.Float[at.Array, "b t d"], 
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
    ]:
        """
        IMLE version with no timestep/noisy action embedding
        """
        input_mask, ar_mask, tokens = [], [], []

        # pass in state token
        state_tok = self.state_proj(obs.state)[:, None, :] # (B,1,E)
        tokens.append(state_tok)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), jnp.bool_))
        ar_mask += [True] # prefix not attending

        # latent action draw
        action_tok = self.action_in_proj(latent_actions) # (B,T,E)
        tokens.append(action_tok)
        input_mask.append(jnp.ones(action_tok.shape[:2], jnp.bool_))
        #  first action token camt see prefix/state,
        #  later ones can attend action sequence
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # concatenate
        tokens      = jnp.concatenate(tokens, axis=1)
        input_mask  = jnp.concatenate(input_mask, axis=1)
        ar_mask     = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    # imle loss
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """
        get prefix from kv cache
        pass through suffix k times for each z we sampled
        then rs 
        """
        preprocess_rng, key_z, key_eps = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        B, T, D = actions.shape # batch, horizon, action_dim
        jax.debug.print("action shape: {}", actions.shape)
        K = self.cfg.num_noise_samples

        # encode prefix, same as sample actions for pi0
        prefix_tok, prefix_m, prefix_ar = self.embed_prefix(observation)
        jax.debug.print("prefix shape: {}", prefix_tok.shape)
        pref_mask = make_attn_mask(prefix_m, prefix_ar)
        jax.debug.print("prefix mask shape: {}", pref_mask.shape)
        pref_pos  = jnp.cumsum(prefix_m, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tok, None], mask=pref_mask, positions=pref_pos
        )

        jax.debug.print("kv_cache shape computed")

        # now pass over each of the k latents, note diff shape 
        z_shape = (K, B, T, D)
        z = jax.random.normal(key_z, z_shape)

        jax.debug.print("z shape: {}", z.shape)

        # vmap over first axis = K
        # alternative is to create a KxB batch and pass it into embed_suffix direclty?
        # reason no is to prevent repeating the kv_cache (and hence increasing memory use).. not 100% sure about this being better tho
        k_acts = jax.vmap(
            lambda z_k: self.sample_actions(
                None,# rng unused
                observation,
                latent=z_k,
                prefix_cache=kv_cache,
                prefix_mask=prefix_m,
            ),
            in_axes=0, # z_k batched, cache broadcast
        )(z)  # (K,B,T,D)

        cand = jnp.transpose(k_acts, (1,0,2,3))

        jax.debug.print("cand shape: {}", cand.shape)


        diff = cand - actions[:, None] # (B, K, T, D)
        if self.cfg.metric == "mse":
            dist = jnp.mean(jnp.square(diff), axis=(-1, -2))
        else:  # "l2"
            dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=(-1, -2)) + 1e-14)

        # rejection sampling
        eps = self.cfg.epsilon
        if isinstance(eps, tuple):
            low, high = eps
            eps = jax.random.uniform(key_eps, (B, 1), minval=low, maxval=high)
        dist = jnp.where(dist < eps, jnp.inf, dist)

        # topk
        if self.cfg.num_backprop == 1:
            loss = dist[jnp.arange(B), jnp.argmin(dist, axis=-1)]
        else:
            loss = jnp.mean(jnp.sort(dist, axis=-1)[:, : self.cfg.num_backprop], axis=-1)

        # avoid inf 
        mask = ~jnp.isinf(loss)
        return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike | None,
        observation: _model.Observation,
        *,
        latent: at.Float[at.Array, "b t d"] | None = None,
        prefix_cache=None,
        prefix_mask=None,
        num_steps: int | at.Int[at.Array, ""] = 1,
    ) -> _model.Actions:
        """
        normal inf:  pass rng, latent/prefix should be None
        loss : pass a pre‑drawn latent
        """
        if latent is None:  # inference
            observation = _model.preprocess_observation(None, observation, train=False)
            B, T, D = observation.state.shape[0], self.action_horizon, self.action_dim
            latent = jax.random.normal(rng, (B, T, D))

            # build prefix once
            prefix_tok, prefix_m, prefix_ar = self.embed_prefix(observation)
            prefix_mask = make_attn_mask(prefix_m, prefix_ar)
            pos = jnp.cumsum(prefix_m, axis=1) - 1
            _, prefix_cache = self.PaliGemma.llm([prefix_tok, None], mask=prefix_mask, positions=pos)

        # suffix pass is shared
        suf_tok, suf_m, suf_ar = self.embed_suffix_imle(observation, latent)
        suf_internal = make_attn_mask(suf_m, suf_ar)
        suf_to_pref  = jnp.repeat(prefix_mask[:, None, :], suf_tok.shape[1], axis=1)
        full_mask    = jnp.concatenate([suf_to_pref, suf_internal], axis=-1)
        pos_suf = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suf_m, axis=1) - 1

        (_, suf_out), _ = self.PaliGemma.llm(
            [None, suf_tok], mask=full_mask, positions=pos_suf, kv_cache=prefix_cache
        )
        return self.action_out_proj(suf_out[:, -self.action_horizon :])

