# Sophia-Pallas
Sophia Pallas is a high-efficiency fine-tuning library designed for the JAX/TPU ecosystem.  While libraries like Unsloth optimize for CUDA, Sophia Pallas leverages the XLA compiler and Pallas kernels to bring "Unsloth-like" memory efficiency to Google Colab TPUs and Cloud TPU v5 pods.  Zero-Dependency on CUDA.  Pure JAX/Flax implementation.

```text
  _________________________________________________________________
 /                                                                 \
|    ███████╗ ██████╗ ██████╗ ██╗  ██╗██╗ █████╗      ██████╗       |
|    ██╔════╝██╔═══██╗██╔══██╗██║  ██║██║██╔══██╗     ██╔══██╗      |
|    ███████╗██║   ██║██████╔╝███████║██║███████║     ██████╔╝      |
|    ╚════██║██║   ██║██╔═══╝ ██╔══██║██║██╔══██║     ██╔═══╝       |
|    ███████║╚██████╔╝██║     ██║  ██║██║██║  ██║     ██║           |
|    ╚══════╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝     ╚═╝           |
|                                                                   |
|             P   A   L   L   A   S    //    T   P   U              |
|                                                                   |
|   "Wisdom is not trained. It is structurally compiled."           |
 \_________________________________________________________________/

   [!] STATUS: EXPERIMENTAL // PRE-ALPHA
   [!] ARCH: JAX / FLAX / PALLAS
   [!] TARGET: TPU v2-8 to TPU v5e
```
### ⚡ Proof of Convergence (TPU v5e)
*Status: Validated on Google Colab TPU Runtime*

Pre-Alpha training run demonstrating successful gradient flow through the Adapter (A/B) matrices while maintaining frozen base weights (W).

| STEP | LOSS       | STATUS        |
|:-----|:-----------|:--------------|
| 0    | 675.42     | 🚀 START      |
| 1    | 515.91     | 📉 CONVERGING |
| ...  | ...        | ...           |
| 9    | 200.70     | ✨ RESONANCE  |

*Optimization Target: Structural Isomorphism in Latent Space.*
