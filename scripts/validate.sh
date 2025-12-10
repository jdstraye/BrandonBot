start=$(date +%s)
pushd ~/workspace/backend
python3.12 -m validation.validator --phase all
popd
end=$(date +%s)
duration=$((end - start))

echo "BrandonBot Validation run Complete (took $duration seconds or $((duration/60)) minutes)."