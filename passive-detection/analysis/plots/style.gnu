# Credit to Charlie Carver for this style sheet :)

reset
set datafile separator ","
set key autotitle columnhead
set terminal postscript eps color enhanced font "Times-Roman" 22
set terminal pdf color enhanced font "Times New Roman, 14"
set encoding iso_8859

# Standard fig size 
set term postscript eps size 4,2.4
set term pdf size  4,2.4

# Key
set key samplen 1.5

# Define colors
#Hex(n) = word("ff4161 57009a 121113 899878", n)
RGBA(h, a) = sprintf("0x%02X%s", 255*(1-a), h)
#RGB(i) = RGBA(i, 1)

# Line width
lw = 2
ps = 0.5

# Solid lines
set style line 1 pt 6 ps ps lw lw dt 1 lc rgb RGBA("0057009a", 0.7)
set style line 2 pt 4 ps ps lw lw dt 1 lc rgb RGBA("00ff4161", 0.7)
set style line 3 pt 16 ps ps lw lw dt 1 lc rgb RGBA("0080ff", 0.7)
set style line 4 pt 12 ps ps lw lw dt 1 lc rgb RGBA("a0a0a0", 0.7)
set style line 5 pt 1 ps ps lw lw dt 1 lc rgb RGBA("f03232", 0.7)
set style line 6 pt 1 ps ps lw lw dt 1 lc rgb RGBA("00c000", 0.7)
