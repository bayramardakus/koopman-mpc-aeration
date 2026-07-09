# How to publish this repository on GitHub and mint a Zenodo DOI

This folder is a complete, ready-to-publish repository. Two options.

## Option A — command line (git)

```bash
cd koopman-mpc-aeration          # this folder
git init
git add -A
git commit -m "Initial release: Koopman-MPC for activated-sludge aeration"
git branch -M main

# create an EMPTY repo on github.com first (no README/License), then:
git remote add origin https://github.com/<your-username>/koopman-mpc-aeration.git
git push -u origin main
```

(If you use the GitHub CLI: `gh repo create koopman-mpc-aeration --public --source=. --push`.)

## Option B — no command line

1. Go to https://github.com/new and create a public repository named
   `koopman-mpc-aeration` (leave "Add a README" unchecked).
2. On the new repo page, click **"uploading an existing file"** and drag in every
   file from this folder (README.md, LICENSE, requirements.txt, CITATION.cff,
   the six `.py` files, and the `results/` folder).
3. Commit.

## Mint the Zenodo DOI (for the manuscript's Data Availability Statement)

1. Sign in at https://zenodo.org with your GitHub account.
2. Go to **Settings → GitHub**, find `koopman-mpc-aeration`, and toggle it **On**.
3. Back on GitHub, create a release: **Releases → Draft a new release**, tag `v1.0.0`,
   title "Initial release", **Publish release**.
4. Zenodo automatically archives the release and issues a DOI
   (e.g., `10.5281/zenodo.1234567`).
5. Replace `10.5281/zenodo.XXXXXXX` in the manuscript and title-page Data Availability
   Statement, and uncomment the `doi:` line in `CITATION.cff`, with the real DOI.

That DOI is what the reviewers and the journal will use to access the code.
