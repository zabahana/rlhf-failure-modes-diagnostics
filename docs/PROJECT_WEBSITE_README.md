# Project Website Deployment

This folder contains a one-page static project website for the paper:

`docs/index.html`

## GitHub Pages

1. Push the repository to GitHub.
2. Open the repository settings.
3. Go to **Pages**.
4. Under **Build and deployment**, choose:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/docs`
5. Save.

GitHub will publish the site at:

`https://zabahana.github.io/rlhf-failure-modes-diagnostics/`

## Custom Domain

To use a custom domain or subdomain, add a `CNAME` file in `docs/` containing the domain name, then configure DNS with your provider.

This site is configured for:

```text
rlhf-paper.zelalem.ai
```

Recommended DNS record:

```text
Type: CNAME
Name: rlhf-paper
Target: zabahana.github.io
```

Use a separate subdomain from the live demo unless you intentionally want this site to replace the demo.
