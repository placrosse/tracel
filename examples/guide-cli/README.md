To run guide-cli using the CLI, use a command of this format:

```sh
cargo run --bin guide-cli -- package --key <API_KEY> --project <PROJECT_ID>
cargo run --bin guide-cli -- run training --functions training --backends wgpu --configs train_configs/config.json --key <API_KEY> --project <PROJECT_ID> --runner <RUNNER_GROUP_NAME>
```