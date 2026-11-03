#!/bin/bash

# The a5xx version of test-regfile.sh. A one-invocation work group never gets
# more than two waves per SP on a5xx, so this launches 64 groups of 256
# invocations instead, and each invocation stores its SP id and wave id with
# stgb. The number of distinct wave ids per SP, as the register footprint
# grows, gives:
# 1. The maximum number of waves per SP.
# 2. The wave granularity, as the step between wave counts.
# 3. The register budget per SP, from the footprints at which the count drops.
#
# Usage: test-regfile-a5xx.sh [single|double]

set -e

wavesize=${1:-single}
width=$([ "$wavesize" = double ] && echo 2 || echo 1)
groups=64
localsize=256

gen_shader() {
	n=$1;
	cat <<EOF
@localsize $localsize, 1, 1
@wavesize $wavesize
@buf $((groups * localsize))  ; g[0]
@wgid(r48.x)
@invocationid(r3.x)
getwid.u32 r1.x
getspid.u32 r1.y
mul.u24 r0.x, r48.x, $localsize
(rpt2)nop
add.u r0.x, r0.x, r3.x
(rpt2)nop
shl.b r2.x, r0.x, 2
mov.u32u32 r2.y, 0

; busy loop to make sure it actually uses all possible waves
mov.u32u32 r0.y, 1024
(rpt2)nop
loop:
cmps.u.gt p0.x, r0.y, 0
sub.u r0.y, r0.y, 1
(rpt5)nop
br p0.x, #loop
add.f r1.z, r1.x, r$n.w

(ss)(sy)(rpt5)nop
shl.b r1.y, r1.y, 16
(rpt2)nop
or.b r1.x, r1.x, r1.y
(rpt5)nop
stgb.untyped.4d.u32.1 g[0], r1.x, r0.x, r2.x
end
nop
EOF
}

# Prints the number of distinct wave ids per SP, then the smallest of them.
count_waves() {
	awk -v n=$((groups * localsize)) '
		function hex(s,  i, v) {
			for (i = 1; i <= length(s); i++)
				v = v * 16 + index("0123456789abcdef", substr(s, i, 1)) - 1
			return v
		}
		/^buf\[0\]:$/ { b = 1; next }
		b && $1 ~ /^[0-9a-f]+$/ && length($1) == 8 {
			for (i = 1; i <= NF && seen < n; i++) {
				seen++
				v = hex($i)
				sp = int(v / 65536)
				if (!((sp, v % 65536) in wid)) {
					wid[sp, v % 65536] = 1
					waves[sp]++
				}
			}
		}
		END {
			min = 0
			for (sp = 0; sp in waves; sp++) {
				printf "sp%d=%d ", sp, waves[sp]
				if (!min || waves[sp] < min)
					min = waves[sp]
			}
			printf "%d\n", min
		}'
}

# w waves of f regs fit the budget and w + g did not, unless w is the cap
budget() {
	awk -v g=$1 -v width=$width '
		{ f[NR] = $1; w[NR] = $2; if ($2 > cap) cap = $2 }
		END {
			for (i = 1; i <= NR; i++) {
				if (w[i] % g)
					exit
				if (w[i] * f[i] * width > lo)
					lo = w[i] * f[i] * width
				u = (w[i] + g) * f[i] * width
				if (w[i] < cap && (!hi || u < hi))
					hi = u
			}
			if (!hi) {
				printf "granularity %d: budget >= %d\n", g, lo
				exit
			}
			if (lo >= hi)
				exit
			printf "granularity %d: %d <= budget < %d,", g, lo, hi
			printf " reg_size_vec4 %d, wave_granularity %d\n", lo / g, g
		}'
}

# Stop at two waves per SP: past that, a footprint that does not fit even one
# group of waves hangs the GPU instead of being refused.
rows=
for n in `seq 3 47`; do
	waves=`gen_shader $n | ./computerator -g $groups,1,1 | count_waves`
	min=${waves##* }
	[ "$min" -gt 0 ] || { echo "r$n: no result"; break; }
	echo "r$n ($((n + 1)) full regs): ${waves% *}"
	rows="$rows$((n + 1)) $min\n"
	[ "$min" -gt 2 ] || break
done

for g in 1 2; do
	printf "$rows" | budget $g
done
