import google.cloud.aiplatform as aip


class VaEncoder:
    
    REQUIRMENTS=[
    "sentence-transformers==2.3.0",
    "transformers==4.36.1",
    "polars==0.20.5",
    "transformers==4.36.1",
    "datasets==2.16.0"
    ]
    def __init__(
        self, 
        gcs_bucket, 
        gcs_prefix,
        project_id,
        region,
        hf_model_name,
        read_data_format,
        write_data_format,
        staging_bucket=None,
        
    ):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.project_id = project_id
        self.region = region
        self.hf_model_name = hf_model_name
        self.read_data_format = read_data_format
        self.write_data_format = write_data_format or read_data_format
        self.staging_bucket = staging_bucket or f'gs://{gcs_bucket}'
        
    def run_job(
        self, 
        args ,
        local=False,
        label=None
        ):
        if not local:
            job = aip.CustomTrainingJob(
            project=self.project_id,
            location=self.region,
            staging_bucket=self.staging_bucket,
            display_name='embed_with_vertex',
            script_path='embed_with_vertex/task.py',
            container_uri="europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
            requirements=self.REQUIRMENTS,
            labels=label
            )
            job = job.run(
            replica_count=1,
            machine_type='n1-highmem-8',
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            base_output_dir=f'gs://{self.gcs_bucket}/{self.gcs_prefix}',
            model_labels=label,
            args=args
        )
        else:
            import subprocess
            args = args + ['--local']
            subprocess.run(['python3', 'embed_with_vertex/task.py'] + args)
    