# Deepseek-from-Scratch
# Without RoPE MLA: Multi-Head Latent Attention

## 📖 Introduction

Transformer-based architectures have revolutionized the fields of NLP, vision, and beyond by enabling scalable and parallelizable attention mechanisms. At the heart of these models lies **Multi-Head Attention (MHA)** — a module that allows models to attend to different parts of the input in parallel using multiple attention "heads."

As transformers evolved, certain computational and memory inefficiencies associated with traditional key-value attention led to the introduction of several optimizations like **KV caching**, **Multi-Query Attention (MQA)**, **Grouped-Query Attention (GQA)**, and now, the proposed **Multi-Head Latent Attention (MHLA)**.

---

## 🎯 Goals of This Module

This repository implements a novel attention mechanism called **Ropeless Multi-Head Latent Attention (Without RoP MLA)**. It is designed to:
- Reduce memory usage by eliminating explicit key projections.
- Reuse a pre-computed key projection pipeline (`absorbed_k`) to simplify attention calculations.
- Maintain head diversity without requiring per-head keys.
- Support efficient incremental decoding via latent KV caching.

---

## 🧠 Background: Multi-Head Attention and Evolution

### 🔹 Standard Multi-Head Attention (MHA)

In the vanilla Transformer, each token is projected into **queries (Q)**, **keys (K)**, and **values (V)**. The attention weights are computed as:

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

This is done because Query, Key, and Value are learned projections of the original embeddings — they help the model learn different roles for tokens in the attention mechanism.

- If we just do dot product of token embeddings, all tokens treat each other the same way.
- But in attention, we want each token to act differently as a query (seeking info) and as a key/value (providing info).
- So we use learned linear layers (W_Q, W_K, W_V) to transform the embeddings into Q, K, V → this gives the model flexibility to learn what to attend to and how.


- **Multi-head** variants split the model dimension `d_model` into `n_heads` subspaces to allow each head to learn distinct semantic relationships.
- However, this architecture has high memory and compute costs — especially with long sequences or during autoregressive inference.
- Also, same computations are repeated for multiple times.
- We have observed that.. the main goal is to predict the next word token. So, for that we only need the last context vector of the sequence

---

### 🔹 The Need for KV Caching

#### 🔍 Problem
During inference (especially for autoregressive generation), recomputing K and V for all tokens at every step is:
- **Redundant**: Past tokens’ keys and values don't change. Repeatative Calculations
- **Inefficient**: Time and memory wasted per decoding step.

#### ✅ Solution
**KV Caching** was introduced:
- Cache the key/value representations of past tokens.
- At each new decoding step, only compute K/V for the current token and append to cache.
- A time complexity change of inferencing occur. i.e, from quadratic drops to linear.

#### ⚠️ Drawbacks
- A huge amount of cache memory is being required due to which other computations get affected.
- Due to this, the inferencing time is reducing but in the cost, we need more data storages. for eg: 400gb in deepseek.

---

### 🔹 Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

To reduce memory footprint:
- **MQA**: Use a single shared set of K and V across all heads.  
  → Fast and low-memory but loses head diversity.
- **GQA**: Use fewer groups of K/V shared among subsets of heads.  
  → Trade-off between efficiency and representational richness.

#### 📉 Limitations
- MQA and GQA degrade in performance on tasks needing high attention granularity (e.g., reasoning, multi-hop).
- Reduces the ability to attend differently across heads.

---

## 💡 Multi-Head Latent Attention (MHLA): Core Idea

MHLA introduces a **latent key-value space**:
- Keys are not computed per token directly.
- Using the latent projection, we calculate the keys and values matrix
- Instead, tokens are projected into a **latent representation** (`latent_dim`), then mapped into key and value spaces via fixed projections.
- Here, the caching size also is reducing as we are caching only one matrix (which was two in prev case) which has lesser dimension than those two.
- So, indeed reducing the size

### ✅ Benefits
- **Head-specific Queries**, but **shared latent space** for keys and values.
- Significantly **reduces memory usage**.
- **Head diversity preserved** using pre-computed absorbed keys.
- **No need to store raw K matrices**.

---

## 🧩 RopelessMLA: Module Overview

### 🔧 Architecture

```
RopelessMLA(d_model=512, n_heads=8, kv_latent_dim=64)
d_model: Full hidden dimension.

n_heads: Number of attention heads.

kv_latent_dim: Dimension of latent key-value space.
```
🔑 Key Components
```
W_q: Projects inputs to query space.

W_dkv: Projects inputs into latent space (shared).

W_uk, W_uv: Projects latent space into keys and values, respectively.

absorbed_k: Pre-computed W_q @ W_uk to avoid redundant computation.
```
🚀 Forward Pass Walkthrough
```
out, kv_cache = model(x, kv_cache=past_kv, past_length=offset)
```
🔄 Step-by-Step
1. Precompute Key Projection (absorbed_k): Compute W_q @ W_uk once. Concept of absorbed query.

2. Latent Encoding: Project input x to latent_dim via W_dkv.

3. Apply LayerNorm for stability.

4. KV Caching: Append new latent codes to kv_cache if decoding incrementally.

5. Value Projection: Project latent to values via W_uv, then reshape to [B, H, S, dh].

6. Query Projection: Reshape input to queries [B, S, H, dh].

7. Attention Score Calculation: Each head uses absorbed_k[h] to map query into latent.

8. Multiply with latent_kvᵀ to get attention scores.

9. Causal Masking: Apply lower-triangular mask for autoregressive attention.

10. Softmax + Weighted Sum:

11. Compute softmax scores and apply to values for output.

12. Concatenate Heads: Final output is [B, S, D].


<table>
  <thead>
    <tr>
      <th>⚖️ Feature</th>
      <th>MHA</th>
      <th>MQA</th>
      <th>GQA</th>
      <th>MHLA (Ours)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Per-head Keys</td>
      <td>✅</td>
      <td>❌</td>
      <td>Partial</td>
      <td>❌ (latent-based)</td>
    </tr>
    <tr>
      <td>Head Diversity</td>
      <td>✅</td>
      <td>❌</td>
      <td>Moderate</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Memory Usage (KV Cache)</td>
      <td>High</td>
      <td>Low</td>
      <td>Moderate</td>
      <td>Low</td>
    </tr>
    <tr>
      <td>Performance (Accuracy)</td>
      <td>High</td>
      <td>Lower</td>
      <td>Balanced</td>
      <td>High</td>
    </tr>
    <tr>
      <td>Best for Long Contexts</td>
      <td>❌</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Inference Speed</td>
      <td>❌</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>
