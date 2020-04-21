#!/bin/bash

# Hyper-params
num_steps=10

# Init classes to gen.
mosque_01=668
lipstick_02=629
brambling_03=10
leafbeetle_04=304
badger_05=362
toaster_06=859
triarch_07=873
cloak_08=501
lawnmower_09=621
library_10=624
cheburger_11=933
swimtrunks_12=842
barn_13=425
candle_14=470
lamp_15=846
sandbar_16=977
frenloaf_17=930
lemon_18=951
chest_19=492
runshoe_20=770
wjug_21=899
ptable_22=736
broom_23=462
cellphone_24=487
craftcarr_25=403
entertain_26=548
jeans_27=608

python synthesize.py --class $mosque_01     --steps $num_steps --sim --gen --cuda --save --name Mosque
python synthesize.py --class $lipstick_02   --steps $num_steps --sim --gen --cuda --save --name Lipstick
python synthesize.py --class $brambling_03  --steps $num_steps --sim --gen --cuda --save --name Brambling
python synthesize.py --class $leafbeetle_04 --steps $num_steps --sim --gen --cuda --save --name LeafBeetle
python synthesize.py --class $badger_05     --steps $num_steps --sim --gen --cuda --save --name Badger
python synthesize.py --class $toaster_06    --steps $num_steps --sim --gen --cuda --save --name Toaster
python synthesize.py --class $triarch_07    --steps $num_steps --sim --gen --cuda --save --name TriumphalArch
python synthesize.py --class $cloak_08      --steps $num_steps --sim --gen --cuda --save --name Cloak
python synthesize.py --class $lawnmower_09  --steps $num_steps --sim --gen --cuda --save --name LawnMower
python synthesize.py --class $library_10    --steps $num_steps --sim --gen --cuda --save --name Library
python synthesize.py --class $cheburger_11  --steps $num_steps --sim --gen --cuda --save --name CheeseBurger
python synthesize.py --class $swimtrunks_12 --steps $num_steps --sim --gen --cuda --save --name SwimmingTrunks
python synthesize.py --class $barn_13       --steps $num_steps --sim --gen --cuda --save --name Barn
python synthesize.py --class $candle_14     --steps $num_steps --sim --gen --cuda --save --name Candle
python synthesize.py --class $lamp_15       --steps $num_steps --sim --gen --cuda --save --name TableLamp
python synthesize.py --class $sandbar_16    --steps $num_steps --sim --gen --cuda --save --name Sandbar
python synthesize.py --class $frenloaf_17   --steps $num_steps --sim --gen --cuda --save --name FrenchLoaf
python synthesize.py --class $lemon_18      --steps $num_steps --sim --gen --cuda --save --name Lemon
python synthesize.py --class $chest_19      --steps $num_steps --sim --gen --cuda --save --name Chest
python synthesize.py --class $runshoe_20    --steps $num_steps --sim --gen --cuda --save --name RunningShoe
python synthesize.py --class $wjug_21       --steps $num_steps --sim --gen --cuda --save --name WaterJug
python synthesize.py --class $ptable_22     --steps $num_steps --sim --gen --cuda --save --name PoolTable
python synthesize.py --class $broom_23      --steps $num_steps --sim --gen --cuda --save --name Broom
python synthesize.py --class $cellphone_24  --steps $num_steps --sim --gen --cuda --save --name Cellphone
python synthesize.py --class $craftcarr_25  --steps $num_steps --sim --gen --cuda --save --name AircraftCarrier
python synthesize.py --class $entertain_26  --steps $num_steps --sim --gen --cuda --save --name EntertainmentCtr
python synthesize.py --class $jeans_27      --steps $num_steps --sim --gen --cuda --save --name Jeans
