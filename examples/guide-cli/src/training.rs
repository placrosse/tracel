use burn::{
    data::dataset::transform::SamplerDataset,
    record::HalfPrecisionSettings,
    train::metric::*,
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use burn::config::Config;

use guide::data::MnistBatcher;
use guide::model::Model;
use guide::training::TrainingConfig;
use tracel::heat::{client::HeatClient, command::DeviceVec, sdk_cli::macros::heat};

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    client: &mut HeatClient,
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) -> Result<Model<B>, ()> {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(SamplerDataset::new(MnistDataset::train(), 100));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(SamplerDataset::new(MnistDataset::test(), 20));

    let recorder =
        tracel::heat::RemoteRecorder::<HalfPrecisionSettings>::checkpoint(client.clone());
    let train_metric_logger = tracel::heat::metrics::RemoteMetricLogger::new_train(client.clone());
    let valid_metric_logger =
        tracel::heat::metrics::RemoteMetricLogger::new_validation(client.clone());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_loggers(train_metric_logger, valid_metric_logger)
        .with_file_checkpointer(recorder)
        .with_application_logger(Some(Box::new(
            tracel::heat::log::RemoteExperimentLoggerInstaller::new(client.clone()),
        )))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    Ok(model_trained)
}

#[heat(training)]
pub fn training<B: AutodiffBackend>(
    mut client: HeatClient,
    config: TrainingConfig,
    DeviceVec(devices): DeviceVec<B::Device>,
) -> Result<Model<B>, ()> {
    train::<B>(&mut client, "/tmp/guide", config, devices[0].clone())
}

#[heat(training)]
pub fn training2<B: AutodiffBackend>(
    config: TrainingConfig,
    DeviceVec(devices): DeviceVec<B::Device>,
    mut client: HeatClient,
) -> Result<Model<B>, ()> {
    train::<B>(&mut client, "/tmp/guide2", config, devices[0].clone())
}


#[heat(training)]
pub fn custom_training<B: AutodiffBackend>(
    DeviceVec(devices): DeviceVec<B::Device>,
) -> Result<Model<B>, ()> {
    println!("Custom training: {:?}", devices);
    Err(())
}

#[heat(training)]
pub fn nothingburger<B: AutodiffBackend>() -> Result<Model<B>, ()> {
    println!("Nothingburger");
    Err(())
}