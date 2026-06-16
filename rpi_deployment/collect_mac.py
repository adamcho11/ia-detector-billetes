#!/usr/bin/env python3
"""Colector Mac - K210 via USB serial. Soporta base64 JPG."""

import serial, time, os, argparse, base64

parser = argparse.ArgumentParser()
parser.add_argument('--port', required=True)
parser.add_argument('--out', default='./fakes_dataset')
parser.add_argument('--baud', type=int, default=115200)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
ser = serial.Serial(args.port, args.baud, timeout=0.1)
ser.reset_input_buffer()
print(f'[{args.port}] Esperando...')

denom='unk'; boxes=[]; jpg=b''; n=0

try:
    while True:
        if ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            if not line: continue

            if line.startswith('CLASIFICA:'):
                denom = line.split(':')[1]
                print(f'\n[{n+1}] {denom} Bs')

            elif line.startswith('DATA:'):
                b64 = line.split(':',1)[1]
                try: jpg = base64.b64decode(b64)
                except: jpg=b''

            elif line.startswith('BOX:'):
                p = line.split(':')
                if len(p)>=6:
                    boxes.append((p[1],p[2],p[3],p[4],p[5]))

            elif line.startswith('END'):
                if jpg:
                    dn = f'{denom}_Bs'
                    fbase = f'{args.out}/{dn}/capture_{n:05d}'
                    os.makedirs(f'{args.out}/{dn}', exist_ok=True)
                    with open(fbase+'.jpg','wb') as f: f.write(jpg)
                    with open(fbase+'.txt','w') as f:
                        f.write(f'DENOM:{denom}\n')
                        for t,x1,y1,x2,y2 in boxes:
                            f.write(f'BOX:{t}:{x1}:{y1}:{x2}:{y2}\n')
                    print(f'  OK {len(jpg)}b {len(boxes)}feat')
                    n+=1
                boxes=[]; jpg=b''

except KeyboardInterrupt:
    print(f'\n[FIN] {n} capturas en {args.out}')
finally:
    ser.close()
