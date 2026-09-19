# Push this commit (auth required on your machine)

Local commit is ready on `/tmp/IMTP` (branch `main`, ahead of origin by 1):

```
110d356 Add selective_news harness for 10-exp H100 runs and paper tables.
```

This environment has **no GitHub credentials** (`Permission denied (publickey)` / HTTPS username prompt fails). Push from a machine where you are logged in:

```bash
# Option A — if this box gets a token
cd /tmp/IMTP
git remote set-url origin https://github.com/CY-H1329/IMTP.git
# PAT with repo scope:
git push https://<GITHUB_USER>:<TOKEN>@github.com/CY-H1329/IMTP.git main

# Option B — SSH key added to the CY-H1329 GitHub account
git remote set-url origin git@github.com:CY-H1329/IMTP.git
git push -u origin main

# Option C — bundle and push elsewhere
cd /tmp/IMTP
git bundle create /tmp/IMTP_selective_news.bundle origin/main..HEAD
# copy bundle to a laptop with auth, then:
#   git pull /path/to/IMTP_selective_news.bundle main
```

After push, on the H100 follow `experiments/selective_news/H100_4GPU_RUN.md`.
