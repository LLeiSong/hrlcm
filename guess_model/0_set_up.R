# Title     : Script to set up
# Objective : To install necessary libraries.
# Created by: Lei Song
# Created on: 03/23/21

########################
##  Step 1: Setting  ##
#######################
message('Setting')

# Define function
package_check <- function(packages){
  invisible(lapply(packages, function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      message(paste('Install package', x))
    } else {
      message(paste('Package', x, 'already was installed.'))
    }
  }))
}

################################
##  Step 2: Install packages  ##
###############################
message('Step 2: Install packages')

## tidyverse
tdv_pkgs <- c("dplyr", "tidyr","stringr")
package_check(tdv_pkgs)

# Spatial
spl_pkgs <- c("sf", "terra","rgrass7")
package_check(spl_pkgs)

# System
sys_pkgs <- c("here", "glue","parallel")
package_check(sys_pkgs)

# Tidymodels
tdm_pkgs <- c("ramify", "ranger","parsnip",
              "tidymodels", "vip", "dials",
              "tune", "ggplot2")
package_check(tdm_pkgs)
