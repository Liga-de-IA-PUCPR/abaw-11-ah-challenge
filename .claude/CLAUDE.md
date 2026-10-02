# Instruções do projeto (Claude Code)

## Branches

- **`develop` é a branch principal do projeto.** Novas branches partem dela e todo PR usa
  `--base develop`.
- **`main` está protegida e congelada** no estado da publicação (artigo do ABAW11): ela precisa
  reproduzir exatamente o resultado publicado (`make reproduce-best`). Nunca commite, faça merge,
  rebase ou abra PR contra a `main`.

## Sessões na nuvem

Setup do cloud environment e do hook de SessionStart: [`references/cloud_environment.md`](../references/cloud_environment.md).
Antes de abrir um PR, rode `make check`.
