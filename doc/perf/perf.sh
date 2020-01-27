pid=${1}
echo "dji" | sudo -S rm *.data
echo "dji" | sudo -S rm out.*
echo "dji" | sudo -S rm *.svg
echo "dji" | sudo -S perf record -g -p ${pid} -F 99 sleep 30
sudo perf script > out.perf
/home/dji/slam/tools/FlameGraph/stackcollapse-perf.pl out.perf > out.perf-folded
/home/dji/slam/tools/FlameGraph/flamegraph.pl out.perf-folded > svo_edgelet.svg
firefox svo_edgelet.svg
echo "dji" | sudo -S chmod 777 perf.data
