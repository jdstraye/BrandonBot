pushd ~/workspace
python3.11 backend/ingest_all.py --chunk-size 128 --overlap 50 ./documents/
popd