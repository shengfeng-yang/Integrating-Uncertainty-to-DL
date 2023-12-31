for s in {9416..9416}
do 
	mkdir $s
	([ -d $s ] && mkdir $s; cd $s;  cp ../Cu.xsf ./;  cp ../change_inverse.py ./;  atomsk Cu.xsf -duplicate 100 100 4  final1.cfg -wrap;  python change_inverse.py)
done
