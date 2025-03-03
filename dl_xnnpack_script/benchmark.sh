for i in ./*.pte; do
	echo $i;
	/home/devuser/dlyakhov/executorch/cmake-out/backends/xnnpack/xnn_executor_runner --model_path  $i --num_iter=100 | tee $i_bench.txt;
done
