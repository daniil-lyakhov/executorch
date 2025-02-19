for i in ./*.pte; do
	echo $i;
	../../../cmake-openvino-out/examples/openvino/openvino_executor_runner --model_path  $i --num_iter=100 | tee $i_bench.txt;
done
