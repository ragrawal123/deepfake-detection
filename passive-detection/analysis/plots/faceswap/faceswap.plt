load "../style.gnu"

set ticscale 0.5
set autoscale xfix
set autoscale yfix

set grid

set xtic format "%.1f"
set xrange [-0.05:1.05]
set xtics 0.25
set xtics nomirror
set xtics in
set xlabel "False Positive Rate"

set ytic format "%.2f"
set yrange [0:1]
set ytics 0.2
set ytics in
set ytics nomirror
set ylabel "True Positive Rate"

set key outside center top
set key maxrows 2
set key width -4
set key samplen 1

out = "faceswap_plot.pdf"
#out2 = "faceswap_plot.eps"
set output out
#set output out2

plot "faceswaprates.csv" using 1:2 w l ls 1 title "EfficientNetAutoAttB4ST - Faceswap (AUC 0.8957)"
