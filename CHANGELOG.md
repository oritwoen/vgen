## [0.3.0] - 2026-03-16

### 🚀 Features

- *(gpu)* Add P2TR support (#7)
- *(gpu)* Add backend support (#8)
- Expose P2SH-P2WPKH as CLI format (#34)
- *(skills)* Add agent skill (#39)

### 🐛 Bug Fixes

- *(csv)* Escape fields containing commas, quotes, or newlines (#22)
- *(pattern)* Validate charset inside character classes (#23)
- *(estimate)* Resolve data provider patterns (#24)
- *(verify)* Derive P2TR, P2SH-P2WPKH, and Ethereum addresses (#25)
- *(range)* Reject --prefix-length 0 for provider patterns (#31)
- *(range)* Fall back to CPU for Ethereum format (#30)
- *(gpu)* Reduce default batch size from 1M to 512K (#32)
- Write TUI results to --file and honor -o format (#33)
- *(pattern)* Validate non-alphanumeric literals inside character classes (#35)
- *(gpu)* Improve error messages for Metal shader failures (#37)

### 💼 Other

- Update wgpu to 28.0 (#10)

### 🚜 Refactor

- *(address)* Add Display and charset_name for AddressFormat (#12)
- Resolve all clippy warnings (#14)
- *(gpu)* Split monolithic shader into separate modules (#36)

### 📚 Documentation

- Standardize badges
- Add Bitcointalk community link
- *(readme)* Add missing address formats and CLI options
- Update AGENTS.md with current codebase state

### ⚙️ Miscellaneous Tasks

- Remove community section from README.md
- Regenerate AGENTS.md
- *(clippy)* Add pedantic and nursery lint config (#29)
## [0.2.0] - 2026-01-04

### 🚀 Features

- *(provider)* Add boha data provider (#2)

### 📚 Documentation

- Add crates.io installation method
- Change yay to paru
- Update AGENTS.md with code map and conventions

### ⚙️ Miscellaneous Tasks

- Add `AGENTS.md`
- Add `deepwiki` badge
- Add `context7.json`
- *(release)* V0.2.0
## [0.1.0] - 2025-12-17

### 🚀 Features

- Initial release

### ⚙️ Miscellaneous Tasks

- *(release)* V0.1.0
