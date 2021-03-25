# Title     : Script for human checking
# Objective : To automatically generate leaflet map
#             for human checking the refined guess labels.
# Created by: Lei Song
# Created on: 03/24/21

#######################
##  Step 1: Setting  ##
#######################
message('Step1: Setting')

library(glue)
library(here)
library(sf)
library(dplyr)
library(stringr)

################################
##  Step 2: Define functions  ##
################################
message('Step 2: Define functions')


####################################
##  Step 3: Pop tiles one by one  ##
####################################
message('Step 3: Pop tiles one by one')
