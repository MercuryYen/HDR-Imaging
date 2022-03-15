pip install -r requirement.txt
python hdr.py -j data\sanctuary\info.json -s 100 -p 500
python alignment.py -j data\exposures\info.json
python tone.py -j data\sanctuary\info.json -s 600 -p 500