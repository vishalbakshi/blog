---
layout: post
description: A tutorial to create a geodatabase, maps and layouts to visualize U.S. Census Data in ArcGIS Pro
categories: [markdown]
title: Visualize U.S. Census Data with ArcGIS
---

By Vishal Bakshi

In this blog post, I’ll walk through my process of creating an ArcGIS geodatabase and a set of layouts visualizing U.S. Census Data. The data used for this app is from table B20005 (Sex By Work Experience In The Past 12 Months By Earnings In The Past 12 Months).

## Table of Contents

- [Get the Data](#get-the-data)
  - [Tract Boundaries](#tract-boundaries-1)
  - [ACS 5-Year Estimates](#acs-5-year-estimates-1)
  - [Using data.census.gov](#using-data-census-gov)
  - [Using the censusapi R package](#using-censusapi)
- [Connect Data to Geodatabase](#connect-data-to-geodatabase)
  - [Tract Boundaries](#tract-boundaries-2)
  - [ACS 5-Year Estimates](#acs-5-year-estimates-2)
- [Visualize Data](#visualize-data)
  - [Create a Map](#create-a-map)
  - [Create a Symbology](#create-a-symbology)
  - [Create a Layout](#create-a-layout)
- [Normalize the Data](#normalize-the-data)
  - [Create Additional Layouts](#create-additional-layouts)

## Get the Data
### Tract Boundaries<a name=""></a>

- Download and unzip 2019 TIGER Shapefile for MN (tl_2019_27_tract.zip) (corresponds to the final year, 2019, in the ACS 5-year estimates). These will contain the Census Tract geographies needed to create a map in ArcGIS.

### ACS 5-Year Estimates<a name=""></a>
#### Using data.census.gov<a name=""></a>

- On data.census.gov, search for B20005

<img src="{{ site.baseurl }}/images/arcgis_01.png" width="50%" />

- Select the link to the Table B20005 with “2019 inflation-adjusted dollars”

<img src="{{ site.baseurl }}/images/arcgis_02.png" width="50%" />

- Click the dropdown at the top next to the label **Product** and select _2015: ACS 5-Year Estimates Detailed Tables_

<img src="{{ site.baseurl }}/images/arcgis_03.png" width="50%" />

- Click **Customize Table** at the top right of the page

<img src="{{ site.baseurl }}/images/arcgis_04.png" width="50%" />

- In the **Geo*** section, click _Tract > Minnesota > All Census Tracts within Minnesota_

<img src="{{ site.baseurl }}/images/arcgis_05.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_06.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_07.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_08.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_09.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_10.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_11.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_12.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_13.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_14.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_15.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_16.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_17.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_18.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_19.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_20.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_21.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_22.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_23.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_24.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_25.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_26.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_27.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_28.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_29.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_30.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_31.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_32.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_33.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_34.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_35.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_36.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_37.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_38.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_39.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_40.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_41.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_42.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_43.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_44.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_45.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_46.png" width="50%" />
<img src="{{ site.baseurl }}/images/arcgis_47.png" width="50%" />

