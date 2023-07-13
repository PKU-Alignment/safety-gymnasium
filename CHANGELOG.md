# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------
## In development
- chore: add CITATION.cff in PR [#53](https://github.com/PKU-Alignment/safety-gymnasium/pull/53).
- chore: update link to PKU-Alignment org in PR [#55](https://github.com/PKU-Alignment/safety-gymnasium/pull/55).
- chore(LICENSE): update license copyright owner.
- feat: support environments wrappers in PR [#57](https://github.com/PKU-Alignment/safety-gymnasium/pull/57).

## [0.4.0] 2023-05-10

- chore(benchmarks): upload benchmark results for environments in PR [#48](https://github.com/PKU-Alignment/safety-gymnasium/pull/48).
- feat(agents): support Doggo agent in PR [#51](https://github.com/PKU-Alignment/safety-gymnasium/pull/51).

## [0.3.0] 2023-04-24

- chore: update link to OmniSafeAi org in PR [#38](https://github.com/PKU-Alignment/safety-gymnasium/pull/38).
- chore(pre-commit): [pre-commit.ci] autoupdate in PR [#36](https://github.com/PKU-Alignment/safety-gymnasium/pull/36).
- deps(gymnasium): update gymnasium version to 0.28.1 in PR [#41](https://github.com/PKU-Alignment/safety-gymnasium/pull/41).
- style: fix grammar in README and normalize string in pyproject.toml.
- chore: update license header in PR [#44](https://github.com/PKU-Alignment/safety-gymnasium/pull/44).
- chore: add test configurations and update Makefile in PR [#45](https://github.com/PKU-Alignment/safety-gymnasium/pull/45).
- feat: add Gymnasium conversion wrappers in PR [#46](https://github.com/PKU-Alignment/safety-gymnasium/pull/46).
- style: prefer utf-8 over UTF-8 in code.
- chore: pre-commit autoupdate.
- test: enable tests in ci in PR [#16](https://github.com/PKU-Alignment/safety-gymnasium/pull/16).

## [0.2.0] 2023-04-05

- feat(SafeVelocity): update safety velocity tasks to v1 in PR [#37](https://github.com/PKU-Alignment/safety-gymnasium/pull/37).
- feat: add ruff and codespell integration in PR [#35](https://github.com/PKU-Alignment/safety-gymnasium/pull/35).

## [0.1.3] 2023-03-26

- fix(builder.py): fix seed setting in PR [#32](https://github.com/PKU-Alignment/safety-gymnasium/pull/32).

## [0.1.2] 2023-03-15

- docs: update example and changelog in PR [#30](https://github.com/PKU-Alignment/safety-gymnasium/pull/30).
- feat: support rgb_array_list and depth_array_list mode for render in PR [#26](https://github.com/PKU-Alignment/safety-gymnasium/pull/26).
- fix: fix bug in vision observation in PR [#28](https://github.com/PKU-Alignment/safety-gymnasium/pull/28).
- chore(vision_env): update default resolution of vision environments in PR [#29](https://github.com/PKU-Alignment/safety-gymnasium/pull/29).

## [0.1.1] 2023-02-27

- fix: fix origin AutoresetWrapper doesn't have cost in PR [#23](https://github.com/PKU-Alignment/safety-gymnasium/pull/23).
- fix: fix the bug in the observation space of the compass in PR [#17](https://github.com/PKU-Alignment/safety-gymnasium/pull/17).
- feat: enable more flake8 checks in PR [#24](https://github.com/PKU-Alignment/safety-gymnasium/pull/24).

## [0.1.0b0] 2023-02-08

### Added

- The first beta release of Safety-Gymnasium.
- Add `task:safe-velocity` and `task:safe-navigation`.
- Add [documentation](www.safety-gymnasium.com).
- Add `robot:racecar` and `robot:ant`.

## [0.1.0] 2023-02-08

### Added

- The first stable release of Safety-Gymnasium.
- Fix confliction of seed mechanism in vector env module.
