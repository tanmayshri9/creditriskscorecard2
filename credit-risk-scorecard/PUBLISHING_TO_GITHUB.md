# How to publish this project to GitHub & share it

A short, copy-paste guide. Pick **Option A** (web upload, no tools) if you're not comfortable with
the command line, or **Option B** (git) if you are.

---

## Before you start (2 minutes)

1. **Your name is already set** to *Tanmay Shrivastava* throughout the project (the README, model
   documentation, source-file headers, and the report). Nothing to edit — but if you ever want to
   change it, search the project for `Tanmay Shrivastava`, and after editing
   `src/build_report.py` re-run `python src/build_report.py` to refresh the report.
2. **Create a free GitHub account** at https://github.com if you don't have one.

---

## Option A — Upload via the GitHub website (no installation)

1. Go to https://github.com/new
2. **Repository name:** `credit-risk-scorecard` · set to **Public** · do **not** tick "Add a README"
   (you already have one) · click **Create repository**.
3. On the next page click **uploading an existing file**.
4. Drag the **entire contents** of the `credit-risk-scorecard` folder into the browser
   (or zip it and drag the zip — GitHub will keep the structure if you drag the folder contents).
   > Tip: select all files/folders *inside* `credit-risk-scorecard`, not the outer folder itself.
5. Add a commit message like `Initial commit — PD credit scorecard` and click **Commit changes**.

Done. Your repo is live at `https://github.com/<your-username>/credit-risk-scorecard`.

---

## Option B — Command line (git)

```bash
# 1. From inside the project folder
cd credit-risk-scorecard

# 2. Initialise and commit
git init
git add .
git commit -m "Initial commit — PD credit scorecard"

# 3. Create the repo on github.com (New repository, name it credit-risk-scorecard, Public).
#    Then connect and push (replace <your-username>):
git branch -M main
git remote add origin https://github.com/<your-username>/credit-risk-scorecard.git
git push -u origin main
```

If prompted for a password, use a **Personal Access Token** (GitHub → Settings → Developer settings →
Personal access tokens), not your account password.

---

## Make the HTML report viewable in a browser (GitHub Pages)

GitHub shows the README automatically, but the HTML report will download rather than render unless you
turn on **GitHub Pages**:

1. Repo → **Settings** → **Pages**.
2. Under **Build and deployment → Source**, choose **Deploy from a branch**.
3. Branch: **main**, folder: **/ (root)** → **Save**.
4. Wait ~1 minute. Your report will be live at:
   ```
   https://<your-username>.github.io/credit-risk-scorecard/outputs/Credit_Risk_Scorecard_Report.html
   ```

That's the single link to put in your CV, LinkedIn, or an email to the recruiter.

---

## Sharing it with the recruiter — what to send

- **The repo:** `https://github.com/<your-username>/credit-risk-scorecard`
  (the README is the landing page — it shows the charts and results immediately).
- **The live report:** the GitHub Pages link above, *or* just attach
  `outputs/Credit_Risk_Scorecard_Report.html` to an email — it's a single self-contained file that
  opens in any browser.

**Suggested one-liner for your application / message:**

> I built an end-to-end probability-of-default credit scorecard to demonstrate the full model
> development lifecycle relevant to this role — WOE/IV feature engineering, a scaled logistic
> scorecard benchmarked against XGBoost, KS/Gini validation, PSI/CSI monitoring, and the lending-
> strategy P&L it drives, all under SR 11-7 governance. Code and a visual report:
> github.com/<your-username>/credit-risk-scorecard

---

## Pin it on your profile

GitHub → your profile → **Customize your pins** → select `credit-risk-scorecard` so it's the first
thing a visitor sees.
