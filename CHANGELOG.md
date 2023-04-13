# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------
## [0.1.2] 2023-03-14

### Added
- feat: support rgb_array_list and depth_array_list mode for render in PR [#26](https://github.com/OmniSafeAI/safety-gymnasium/pull/26).
- fix: fix bug in vision observation in PR [#28](https://github.com/OmniSafeAI/safety-gymnasium/pull/28).
- chore(vision_env): update default resolution of vision environments in PR [#29](https://github.com/OmniSafeAI/safety-gymnasium/pull/29).

## [0.1.1] 2023-02-27

### Added
- fix: fix origin AutoresetWrapper doesn't have cost in PR [#23](https://github.com/OmniSafeAI/safety-gymnasium/pull/23).
- fix: fix the bug in the observation space of the compass in PR [#17](https://github.com/OmniSafeAI/safety-gymnasium/pull/17).
- feat: enable more flake8 checks in PR [#24](https://github.com/OmniSafeAI/safety-gymnasium/pull/24).

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
