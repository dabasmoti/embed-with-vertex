from embed_with_vertex.vertex_encoder import VaEncoder





if __name__ == '__main__':
    
    encoder = VaEncoder(
        gcs_bucket='htz-analysts-data',
        gcs_prefix='test',
        project_id='htz-data',
        region='europe-west1',
        hf_model_name='MPA/sambert',
        read_data_format='parquet',
        write_data_format='parquet',
        staging_bucket='gs://htz-analysts-data',
    )
    encoder.run_job(args=[
        '--data_path=gs://htz-nlp/temp_data', 
        '--model_path=MPA/sambert', 
        '--output_path=gs://htz-nlp/temp_data', 
        '--batch_size=32', 
        '--max_seq_length=512', 
        '--text_col=text'
        ],
    local=False)