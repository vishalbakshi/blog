---
layout: post
description: An explanation of my development process for a census data shiny app
categories: [markdown]
title: R Shiny Census App
---
# R Shiny Census App

In this blog post, I'll walk through my development process for a [U.S. Census data visualization web app](http://vbakshi.shinyapps.io/census-app) I created using the Shiny package in R.

## Background

I started this project by reading the handbook <a href="https://www.census.gov/content/dam/Census/library/publications/2020/acs/acs_state_local_handbook_2020.pdf">Understanding and Using American Community Survey Data: What State and Local Government Users Need to Know</a> published by the U.S. Census Bureau. I recreated the handbook's first case study in R, in which they make comparisons across geographic areas, create custom geographic areas from census tracts and calculate margins of error for derived estimates for Minnesota Census Tract 5-year earnings estimates. 

During the process of recreating the derived median earnings estimate calculations, I was unable to recreate a key value from the handbook (the Standard Error for the 50% proportion, calculated to be 0.599) because I was unable to deduce the values used in the following formula referenced from page 17 of the <a href="https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2015_2019AccuracyPUMS.pdf">PUMS Accuracy of the Data documentation</a>:

![Standard Error equals Design Factor times square root of the product of 95 over 5B and 50 squared](images/se_formula_original.png)

The documentation defines B as the _base_, which is the _calculated weighted total_. I chose the value of 1.3 for the design factor DF since it corresponds to STATE = Minnesota, CHARTYP = Population, CHARACTERISTIC = Person Earnings/Income in the <a href="https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2019_PUMS_5yr_Design_Factors.csv">Design Factors CSV published by the Census Bureau</a>.

I called the <a href="https://www.census.gov/programs-surveys/acs/contact.html">Census Bureau Customer Help Center</a> for assistance and was transferred to a member of the ACS Data User Support team with whom I discussed my woes. He was unable to confirm the values of the design factor DF or B, and was unable to pull up the contact information for the statistical methodology team, so I emailed him my questions. After a few email exchanges, the statistical methodoloy team provided the following:

- DF = 1.3
- B = the total population estimate for which the median is being calculated, which is 82488 for the case study calculation (Minnesota Rural Male Full Time Workers)
- The term 95/5 is associated with the finite population correction factor (100 - f) divided by the sample fraction (f), where f = 5% (later on I note in the documentation that this 95/5 term is based on a 68% confidence interval). The data used in the handbook case study is from 5-year estimates. 1-year estimates sample 2.5% of the population, so the 5-year estimates represent a 5 * 2.5 = 12.5% sample. Instead of 95/5, the ratio becomes (100 - 12.5)/12.5 = 87.5/12.5

The updated formula is then:

![Standard Error equals Design Factor times square root of the product of 87.5 over 12.5B and 50 squared](images/se_formula_modified.png)

I was able to calculate the median earnings estimate (and associated standard error and margin of error) within a few percent of the values given in the handbook. This provided me with confirmation that I was ready to expand my code to calculate median earnings estimates for other subgroups.

## The Stack

I built this app using the R package <a href="https://shiny.rstudio.com/reference/shiny/latest/">`Shiny`</a> which handles both the UI and the server. I stored the data in a `sqlite` database and accessed it with queries written using the <a href="https://cran.r-project.org/web/packages/RSQLite/RSQLite.pdf">`RSQLite`</a> package which uses the <a href="https://dbi.r-dbi.org/reference/">DBI</a> API. The following section breaks down the R scripts based on functionality. You click on the script name to navigate to that section.

## The Codebase

- [`app.R`](#app-r)
  - UI and server functions to handle people inputs and plot/table/text outputs
- [`prep_db.R`](#prep-db-r)
  - Import, clean, combine and then load data into the `census_app_db.sqlite` database
- [`get_b20005_ruca_aggregate_earnings.R`](#get-b20005-ruca-aggregate-earnings-r)
  - Receives input values from UI and calls helper functions to query the database for the necessary data
- [`get_{sex}_{work_status}_ruca_aggregate_b20005_earnings.R`](#get-b20005-sex-work_status-ruca-aggregate-earnings-r)
  - Queries the SQLite database for population estimates of workers of the given sex and work status, aggregated by RUCA levels for a given state
- [`get_b20005_earnings.R`](#get-b20005-earnings)
  - Receives input values from UI and calls helper functions to query the database for the necessary data
- [`get_{sex}_{work_status}_b20005_earnings.R`](#get-b20005-sex-work_status-earnings-r)
  - Queries the SQLite database for population estimates of workers of the given sex and work status for each census tract in the given state
- [`get_all_b20005_earnings.R`](#get-all-b20005-earnings-r)
  - Queries the SQLite database for population estimates for all sexes and work statuses for each census tract in the state
- [`get_b20005_states.R`](#get-b20005-states-r)
  - Queries the SQLite database for a list of U.S. states 
- [`get_design_factor.R`](#get-design-factor-r)
  - Receives state selection from UI and queries database for the design factor used for the median earnings estimation calculation
- [`get_b20005_labels.R`](#get-b20005-labels-r)
  - Queries the database for descriptive labels of B20005 table variables
- [`format_query_result.R`](#format-query-result-r)
  - Receives the query result `data.frame` of earnings data and makes it prettier before it's displayed on the UI
- [`calculate_median.R`](#calculate-median-r)
  - Receives population estimates for earnings levels aggregated by RUCA level and the design factor and returns the median earnings estimate, standard of error and margin of error
- [`make_plot.R`](#make-plot-r)
  - Receives earnings `data.frame` and RUCA level selected from UI and returns a bar plot

### <a name="app-r"></a>`app.R`
### <a name="prep-db-r"></a>`prep_db.R`
### <a name="get-b20005-ruca-aggregate-earnings-r"></a>`get_b20005_ruca_aggregate_earnings.R`
### <a name="get-b20005-sex-work_status-ruca-aggregate-earnings-r"></a>`get_b20005_{sex}_{work_status}_ruca_aggregate_earnings.R`
### <a name="get-b20005-earnings-r"></a>`get_b20005_earnings.R`
### <a name="get-b20005-sex-work_status-earnings-r"></a>`get_b20005_{sex}_{work_status}_earnings.R`
### <a name="get-all-b20005-earnings-r"></a>`get_all_b20005_earnings.R`
### <a name="get-b20005-states-r"></a>`get_b20005_earnings.R`
### <a name="get-design-factor-r"></a>`get_design_factor.R`
### <a name="get-b20005-labels-r"></a>`get_b20005_labels.R`
### <a name="format-query-result-r"></a>`format_query_result.R`
### <a name="calculate-median-r"></a>`calculate_median.R`
### <a name="make-plot-r"></a>`make_plot.R`

