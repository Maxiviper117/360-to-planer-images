
# Using tool:
### [Bump My Version](https://github.com/callowayproject/bump-my-version)

#### [Docs](https://callowayproject.github.io/bump-my-version/)



Show the current version

```bash
bump-my-version show current_version
```

Performing bump dry run

```bash
bump-my-version bump patch --dry-run -vv   # verbose
```

Performing bump

```bash
bump-my-version bump patch 
```

This will update  the [`VERSION`](./VERSION) file and the [`.bumpversion.toml`](./.bumpversion.toml)