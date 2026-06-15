# quant-fundamentals

Stochastic finance models implemented from scratch — MSc Quantitative Finance, Bayes Business School.

**Live site → [your-username.github.io/quant-fundamentals](https://github.com)**

## Models

| File | Model | SDE |
|------|-------|-----|
| `finance/ito.py` | Itô process | $dX = b\,dt + \sigma\,dW$ |
| `finance/gbm.py` | Geometric Brownian motion | $dS = \mu S\,dt + \sigma S\,dW$ |
| `finance/cir.py` | CIR short-rate model | $dr = (\alpha - \beta r)\,dt + \gamma\sqrt{r}\,dW$ |
| `finance/heston.py` | Heston stochastic vol | $dV = \kappa(\bar{v}-V)\,dt + \eta\sqrt{V}\,dW$ |
| `finance/levy.py` | Lévy jump-diffusion | $X_t = \mu t + \sigma W_t + \sum J_i$ |
| `finance/var.py` | Value at Risk | historical · rolling · parametric |
| `finance/mv_hedge.py` | Mean-variance hedging | VOMM optimal delta |

## Quick start

```bash
git clone https://github.com/your-username/quant-fundamentals
cd quant-fundamentals
pip install numpy scipy matplotlib
python -c "from finance import gbm; S = gbm.simulate(100, 0.05, 0.2); print(S.shape)"
```

## Hosting on GitHub Pages

1. Push this repo to GitHub
2. Go to **Settings → Pages → Source → main branch → / (root)**
3. Your site will be live at `https://your-username.github.io/quant-fundamentals`

## Stack

- Python 3.10+, NumPy, SciPy
- No external quant libraries — everything built from the SDE up
