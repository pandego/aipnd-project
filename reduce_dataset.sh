for split in train valid test; do
    for class_dir in flowers/$split/*; do
        [ -d "$class_dir" ] && (find "$class_dir" -type f -name "*.jpg" | sort | tail -n +4 | xargs rm -f)
    done
done