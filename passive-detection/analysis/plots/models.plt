load "./style.gnu"

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
set yrange [-0.05:1.05]
set ytics 0.25
set ytics in
set ytics nomirror
set ylabel "True Positive Rate"

set key outside center top
set key maxrows 3
set key width -1
set key samplen 1

#out = "models_roc.pdf"
#set output out

out2 = "models_roc.eps"
set output out2

#set label at 0,0.97 "" point pointtype 4 pointsize 0.5 lw 2 lc rgb RGBA("00ff4161", 0.7)
#set label at 0,0.925 "" point pointtype 4 pointsize 0.5 lw 2 lc rgb RGBA("00ff4161", 0.7)
#set label at 0,0.88 "" point pointtype 4 pointsize 0.5 lw 2 lc rgb RGBA("00ff4161", 0.7)
#set label at 0,0.835 "" point pointtype 4 pointsize 0.5 lw 2 lc rgb RGBA("00ff4161", 0.7)

#set label at 0,0.955  "" point pointtype 6 pointsize 0.5 lw 2 lc rgb RGBA("0057009a", 0.7)
#set label at 0,0.91 "" point pointtype 6 pointsize 0.5 lw 2 lc rgb RGBA("0057009a", 0.7)
#set label at 0,0.865 "" point pointtype 6 pointsize 0.5 lw 2 lc rgb RGBA("0057009a", 0.7)
#set label at 0,0.82 "" point pointtype 6 pointsize 0.5 lw 2 lc rgb RGBA("0057009a", 0.7)

#set label at 0,0.94 "" point pointtype 16 pointsize 0.5 lw 2 lc rgb RGBA("0080ff", 0.7)
#set label at 0,0.895 "" point pointtype 16 pointsize 0.5 lw 2 lc rgb RGBA("0080ff", 0.7)
#set label at 0,0.85 "" point pointtype 16 pointsize 0.5 lw 2 lc rgb RGBA("0080ff", 0.7)
#set label at 0,0.805 "" point pointtype 16 pointsize 0.5 lw 2 lc rgb RGBA("0080ff", 0.7)

plot\
    "./dagan/daganrates.csv" every 27 using 1:2 w lp ls 1 title "Dagan (AUC 0.7142)",\
    "./faceswap/faceswaprates.csv" every 30 using 1:2 w lp ls 2 title "Faceswap (AUC 0.8957)",\
    "./first/firstrates.csv" every 28 using 1:2 w lp ls 3 title "First (AUC 0.8136)",\
    "./sadtalker/sadtalkerrates.csv" every 27 using 1:2 w lp ls 4 title "Sadtalker (AUC 0.7206)",\
    "./talklip/talkliprates.csv" every 30 using 1:2 w lp ls 6 title "Talklip (AUC 0.9568)",\
