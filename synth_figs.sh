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
python synthesize.py --cl $mosque_01     --steps $num_steps --sim --gen --cuda --save --na Mosque

echo Starting job 02/27 -- Lipstick
python synthesize.py --cl $lipstick_02   --steps $num_steps --sim --gen --cuda --save --na Lipstick

echo Starting job 03/27 -- Brambling
python synthesize.py --cl $brambling_03  --steps $num_steps --sim --gen --cuda --save --na Brambling

echo Starting job 04/27 -- LeafBeetle
python synthesize.py --cl $leafbeetle_04 --steps $num_steps --sim --gen --cuda --save --na LeafBeetle

echo Starting job 05/27 -- Badger
python synthesize.py --cl $badger_05     --steps $num_steps --sim --gen --cuda --save --na Badger

echo Starting job 06/27 -- Toaster
python synthesize.py --cl $toaster_06    --steps $num_steps --sim --gen --cuda --save --na Toaster

echo Starting job 07/27 -- TriumphalArch
python synthesize.py --cl $triarch_07    --steps $num_steps --sim --gen --cuda --save --na TriumphalArch

echo Starting job 08/27 -- Cloak
python synthesize.py --cl $cloak_08      --steps $num_steps --sim --gen --cuda --save --na Cloak

echo Starting job 09/27 -- LawnMower
python synthesize.py --cl $lawnmower_09  --steps $num_steps --sim --gen --cuda --save --na LawnMower

echo Starting job 10/27 -- Library
python synthesize.py --cl $library_10    --steps $num_steps --sim --gen --cuda --save --na Library

echo Starting job 11/27 -- CheeseBurger
python synthesize.py --cl $cheburger_11  --steps $num_steps --sim --gen --cuda --save --na CheeseBurger

echo Starting job 12/27 -- SwimmingTrunks
python synthesize.py --cl $swimtrunks_12 --steps $num_steps --sim --gen --cuda --save --na SwimmingTrunks

echo Starting job 13/27 -- Barn
python synthesize.py --cl $barn_13       --steps $num_steps --sim --gen --cuda --save --na Barn

echo Starting job 14/27 -- Candle
python synthesize.py --cl $candle_14     --steps $num_steps --sim --gen --cuda --save --na Candle

echo Starting job 15/27 -- TableLamp
python synthesize.py --cl $lamp_15       --steps $num_steps --sim --gen --cuda --save --na TableLamp

echo Starting job 16/27 -- Sandbar
python synthesize.py --cl $sandbar_16    --steps $num_steps --sim --gen --cuda --save --na Sandbar

echo Starting job 17/27 -- FrenchLoaf
python synthesize.py --cl $frenloaf_17   --steps $num_steps --sim --gen --cuda --save --na FrenchLoaf

echo Starting job 18/27 -- Lemon
python synthesize.py --cl $lemon_18      --steps $num_steps --sim --gen --cuda --save --na Lemon

echo Starting job 19/27 -- Chest
python synthesize.py --cl $chest_19      --steps $num_steps --sim --gen --cuda --save --na Chest

echo Starting job 20/27 -- RunningShoe
python synthesize.py --cl $runshoe_20    --steps $num_steps --sim --gen --cuda --save --na RunningShoe

echo Starting job 21/27 -- WaterJug
python synthesize.py --cl $wjug_21       --steps $num_steps --sim --gen --cuda --save --na WaterJug

echo Starting job 22/27 -- PoolTable
python synthesize.py --cl $ptable_22     --steps $num_steps --sim --gen --cuda --save --na PoolTable

echo Starting job 23/27 -- Broom
python synthesize.py --cl $broom_23      --steps $num_steps --sim --gen --cuda --save --na Broom

echo Starting job 24/27 -- Cellphone
python synthesize.py --cl $cellphone_24  --steps $num_steps --sim --gen --cuda --save --na Cellphone

echo Starting job 25/27 -- AircraftCarrier
python synthesize.py --cl $craftcarr_25  --steps $num_steps --sim --gen --cuda --save --na AircraftCarrier

echo Starting job 26/27 -- EntertainmentCtr
python synthesize.py --cl $entertain_26  --steps $num_steps --sim --gen --cuda --save --na EntertainmentCtr

echo Starting job 27/27 -- Jeans
python synthesize.py --cl $jeans_27      --steps $num_steps --sim --gen --cuda --save --na Jeans


# Plot it.
python synth_plot.py
