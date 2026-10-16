# quillsync

`quillsync` keeps a folder of Markdown notes synchronised between a laptop and a
self-hosted server. It is written in Go and ships as a single binary.

## Installation

### From a release

Download the archive for your platform and put the binary on your `PATH`:

```bash
curl -LO https://example.org/quillsync/v2.4.1/quillsync-linux-amd64.tar.gz
tar -xzf quillsync-linux-amd64.tar.gz
sudo mv quillsync /usr/local/bin/
```

The current stable release is **version 2.4.1**, published on 3 March 2026.

### From source

Building from source requires Go 1.22 or newer:

```bash
git clone https://example.org/quillsync.git
cd quillsync
make build
```

## Configuration

The configuration file lives at `~/.config/quillsync/config.yaml`:

```yaml
server: https://notes.example.org
folder: ~/notes
interval: 300
conflict_policy: keep-both
```

| Option | Default | Meaning |
|---|---|---|
| `server` | none | URL of the sync server |
| `folder` | `~/notes` | Local folder to synchronise |
| `interval` | `300` | Seconds between automatic syncs |
| `conflict_policy` | `keep-both` | What to do when both sides changed |

## Commands

### sync

Runs one synchronisation immediately. With `--dry-run` it prints the planned uploads and
downloads without touching any file.

### status

Shows the last successful sync time, the number of pending changes, and whether the
server is reachable.

### prune

Deletes local backup copies older than 30 days. Use `--keep 90` to keep backups for 90
days instead.

## Troubleshooting

- **"certificate signed by unknown authority"**: add your server's CA certificate with
  `quillsync trust ca.pem`.
- **Sync never starts**: check that `interval` is a positive number of seconds.
- **Duplicate files ending in `.conflict`**: these appear when `conflict_policy` is
  `keep-both`. Merge them by hand and delete the copy.

## Contributing

Pull requests are welcome. Run `make test` before opening one, and add a line to the
changelog. Every new command needs a section in this README.

## License

quillsync is released under the MIT License. Copyright 2024-2026 the quillsync authors.
