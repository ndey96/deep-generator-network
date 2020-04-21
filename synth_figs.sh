#!/bin/bash

# Hyper-params
num_steps=100

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

# Run all the jobs.
echo Starting job 01/27 -- Mosque
python synthesize.py --class $mosque_01     --steps $num_steps --sim --gen --cuda --save --name Mosque

echo Starting job 02/27 -- Lipstick
python synthesize.py --class $lipstick_02   --steps $num_steps --sim --gen --cuda --save --name Lipstick

echo Starting job 03/27 -- Brambling
python synthesize.py --class $brambling_03  --steps $num_steps --sim --gen --cuda --save --name Brambling

echo Starting job 04/27 -- LeafBeetle
python synthesize.py --class $leafbeetle_04 --steps $num_steps --sim --gen --cuda --save --name LeafBeetle

echo Starting job 05/27 -- Badger
python synthesize.py --class $badger_05     --steps $num_steps --sim --gen --cuda --save --name Badger

echo Starting job 06/27 -- Toaster
python synthesize.py --class $toaster_06    --steps $num_steps --sim --gen --cuda --save --name Toaster

echo Starting job 07/27 -- TriumphalArch
python synthesize.py --class $triarch_07    --steps $num_steps --sim --gen --cuda --save --name TriumphalArch

echo Starting job 08/27 -- Cloak
python synthesize.py --class $cloak_08      --steps $num_steps --sim --gen --cuda --save --name Cloak

echo Starting job 09/27 -- LawnMower
python synthesize.py --class $lawnmower_09  --steps $num_steps --sim --gen --cuda --save --name LawnMower

echo Starting job 10/27 -- Library
python synthesize.py --class $library_10    --steps $num_steps --sim --gen --cuda --save --name Library

echo Starting job 11/27 -- CheeseBurger
python synthesize.py --class $cheburger_11  --steps $num_steps --sim --gen --cuda --save --name CheeseBurger

echo Starting job 12/27 -- SwimmingTrunks
python synthesize.py --class $swimtrunks_12 --steps $num_steps --sim --gen --cuda --save --name SwimmingTrunks

echo Starting job 13/27 -- Barn
python synthesize.py --class $barn_13       --steps $num_steps --sim --gen --cuda --save --name Barn

echo Starting job 14/27 -- Candle
python synthesize.py --class $candle_14     --steps $num_steps --sim --gen --cuda --save --name Candle

echo Starting job 15/27 -- TableLamp
python synthesize.py --class $lamp_15       --steps $num_steps --sim --gen --cuda --save --name TableLamp

echo Starting job 16/27 -- Sandbar
python synthesize.py --class $sandbar_16    --steps $num_steps --sim --gen --cuda --save --name Sandbar

echo Starting job 17/27 -- FrenchLoaf
python synthesize.py --class $frenloaf_17   --steps $num_steps --sim --gen --cuda --save --name FrenchLoaf

echo Starting job 18/27 -- Lemon
python synthesize.py --class $lemon_18      --steps $num_steps --sim --gen --cuda --save --name Lemon

echo Starting job 19/27 -- Chest
python synthesize.py --class $chest_19      --steps $num_steps --sim --gen --cuda --save --name Chest

echo Starting job 20/27 -- RunningShoe
python synthesize.py --class $runshoe_20    --steps $num_steps --sim --gen --cuda --save --name RunningShoe

echo Starting job 21/27 -- WaterJug
python synthesize.py --class $wjug_21       --steps $num_steps --sim --gen --cuda --save --name WaterJug

echo Starting job 22/27 -- PoolTable
python synthesize.py --class $ptable_22     --steps $num_steps --sim --gen --cuda --save --name PoolTable

echo Starting job 23/27 -- Broom
python synthesize.py --class $broom_23      --steps $num_steps --sim --gen --cuda --save --name Broom

echo Starting job 24/27 -- Cellphone
python synthesize.py --class $cellphone_24  --steps $num_steps --sim --gen --cuda --save --name Cellphone

echo Starting job 25/27 -- AircraftCarrier
python synthesize.py --class $craftcarr_25  --steps $num_steps --sim --gen --cuda --save --name AircraftCarrier

echo Starting job 26/27 -- EntertainmentCtr
python synthesize.py --class $entertain_26  --steps $num_steps --sim --gen --cuda --save --name EntertainmentCtr

echo Starting job 27/27 -- Jeans
python synthesize.py --class $jeans_27      --steps $num_steps --sim --gen --cuda --save --name Jeans


# Plot it.
python synth_plot.py
