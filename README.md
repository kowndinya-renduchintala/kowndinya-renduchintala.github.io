# kowndinya-renduchintala.github.io

Source for my personal website, [kowndinya-renduchintala.github.io](https://kowndinya-renduchintala.github.io).

## Where things live

| What                        | Where                                                    |
| --------------------------- | -------------------------------------------------------- |
| About page (home)           | `_pages/about.md`                                        |
| Blog posts                  | `_posts/` (per-post citations in `assets/bibliography/`) |
| News / announcements        | `_news/`                                                 |
| Publications                | `_bibliography/papers.bib`                               |
| Gallery, repositories, etc. | `_pages/`                                                |
| Images and PDFs             | `assets/img/`, `assets/pdf/`                             |
| Site settings               | `_config.yml`                                            |

Pushing to `master` builds and deploys the site through `.github/workflows/deploy.yml`.

## Running locally

With Docker:

```bash
docker compose pull
docker compose up
```

Then open <http://localhost:8080>.

Or with Ruby installed:

```bash
bundle install
bundle exec jekyll serve
```

## Credits

Built with [Jekyll](https://jekyllrb.com/) and the [al-folio](https://github.com/alshedivat/al-folio) theme (MIT license, see `LICENSE`).
For theme documentation, see al-folio's [INSTALL](https://github.com/alshedivat/al-folio/blob/main/INSTALL.md), [CUSTOMIZE](https://github.com/alshedivat/al-folio/blob/main/CUSTOMIZE.md), and [FAQ](https://github.com/alshedivat/al-folio/blob/main/FAQ.md) guides.
