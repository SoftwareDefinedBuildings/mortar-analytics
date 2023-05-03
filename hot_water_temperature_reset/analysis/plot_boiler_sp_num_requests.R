# Visualize data from testing the hot water supply temperature
# reset strategy at DBC.
# 
# @author Carlos Duarte <cduarte@berkeley.edu>

library(tidyverse)
library(lubridate)
library(ggplot2)
library(scales)


usr_folder <- file.path('C:','Users', 'duar3')
github_folder <- file.path(usr_folder, 'Documents', 'github')

fig_folder <- file.path('./', 'figures')

source(file.path(github_folder, 'publication_formats', 'r_ggplot_formats.R'))


IPtoSI_temp_conversion <- function(degF) {
  return((degF-32)/1.8)
}

coeff = 15

fig_type <- 'presentation_plot_area'
fig_type <- 'graphical_abstract_grid'
plot_type <- 'png'
fig_size_factor <- 1/2

dec_dates <- ymd_hm(c('2021-12-01 0:00', '2021-12-03 23:59'), tz='America/Los_Angeles')
dec_plt_dates <- ymd_hm(c('2021-12-7 0:00', '2021-12-10 23:59'), tz='America/Los_Angeles')

mar_dates <- ymd_hm(c('2022-03-01 0:00', '2022-03-31 23:59'), tz='America/Los_Angeles')
mar_plt_dates <- ymd_hm(c('2022-03-14 0:00', '2022-03-17 23:59'), tz='America/Los_Angeles')

month_dates <- dec_dates
plt_dates <- dec_plt_dates

ignore_req <- 2

num_requests <- read_csv('./DATA/number_of_request.csv', 
                         col_names = c('req_num_fast_react', 'req_num_slow_react', 'Total Requests', 'Timestamp'),
                         ) %>%
  select(Timestamp, `Total Requests`) %>%
  mutate(Timestamp = ymd_hms(Timestamp, tz='America/Los_Angeles'))
  # rename(value = `number of request sent to fmu`) %>%
  # mutate(metric = 'number of requests',
  #        value = value*coeff,
  #        timestamp = mdy_hm(timestamp))

## december average
month_calc <- num_requests %>%
  filter(Timestamp >= dec_dates[1],
         Timestamp <= dec_dates[2]) %>%
  mutate(total_minutes = c(NaN, diff(Timestamp)/60)) %>%
  filter(!is.na(total_minutes))

sum(month_calc$`Total Requests`*month_calc$total_minutes)/sum(month_calc$total_minutes)

## march average
month_calc <- num_requests %>%
  filter(Timestamp >= mar_dates[1],
         Timestamp <= mar_dates[2]) %>%
  mutate(total_minutes = c(NaN, diff(Timestamp)/60)) %>%
  filter(!is.na(total_minutes))

sum(month_calc$`Total Requests`*month_calc$total_minutes)/sum(month_calc$total_minutes)

## December plot average
plot_calc <- num_requests %>%
  filter(Timestamp >= dec_plt_dates[1],
         Timestamp <= dec_plt_dates[2]) %>%
  mutate(total_minutes = c(NaN, diff(Timestamp)/60)) %>%
  filter(!is.na(total_minutes))

sum(plot_calc$`Total Requests`*plot_calc$total_minutes)/sum(plot_calc$total_minutes)

## March plot average
plot_calc <- num_requests %>%
  filter(Timestamp >= mar_plt_dates[1],
         Timestamp <= mar_plt_dates[2]) %>%
  mutate(total_minutes = c(NaN, diff(Timestamp)/60)) %>%
  filter(!is.na(total_minutes))

sum(plot_calc$`Total Requests`*plot_calc$total_minutes)/sum(plot_calc$total_minutes)

boiler_setpoint_cdl_calc <- read_csv('./DATA/boiler_setpoint.csv',
                            col_names = c('Timestamp', 'New Controller Setpoint [K]')) %>%
  mutate(`New G36 Controller Setpoint` = (`New Controller Setpoint [K]` - 273.15)*1.8 + 32,
         Timestamp = ymd_hms(Timestamp, tz='America/Los_Angeles')) %>%
  pivot_longer(!Timestamp, names_to = "stream_names", values_to = "temps")
  
  # mutate(metric = 'boiler setpoint [F]',
  #        value = (value - 273.15)*1.8 + 32,
  #        timestamp = mdy_hm(timestamp)
  #        )


boiler_temps_bacnet <- read_csv('./DATA/dec_2021_smap_boiler_temps.csv') %>%
  rename(`Hot Water Supply` = `Field Bus1.Plant_Boilers.HWS-T`,
         `Hot Water Return` = `Field Bus1.Plant_Boilers.HWR-T`,
         `Boiler Supply Setpoint` = `Field Bus1.Plant_Boilers.HWS-TEffSetpnt`
         ) %>%
  mutate(Timestamp = with_tz(Timestamp, 'America/Los_Angeles')) %>%
  pivot_longer(!Timestamp, names_to = "stream_names", values_to = "temps") %>%
  drop_na(temps)

## plot average
plt_boil_temps <- boiler_temps_bacnet %>%
  filter(Timestamp >= plt_dates[1],
         Timestamp <= plt_dates[2]) %>%
  group_by(stream_names) %>%
  summarise(temps = mean(temps))

plt_boil_temps
# convert to SI
unlist(lapply(plt_boil_temps$temps, IPtoSI_temp_conversion))

# Request Plot
req_plot <- num_requests %>%
  filter(Timestamp >= plt_dates[1],
         Timestamp <= plt_dates[2]) %>%
  ggplot(., aes(Timestamp, `Total Requests`)) +
  fig_format[[fig_type]][["fig_format"]] +
  geom_step(linewidth=1.25*fig_size_factor, color='#3ed579') +
  geom_hline(yintercept=ignore_req, linewidth=1.5*fig_size_factor, linetype='dashed', color='#d5793e') +
  annotate('text', x=plt_dates[1] + hours(4), y=ignore_req+.7, label='Ignored Requests', size=6*fig_size_factor, color='#d5793e') +
  theme(axis.title.x=element_blank()) +
  scale_x_datetime(date_breaks = "1 day", labels=date_format("%b %d")) +
  scale_y_continuous(breaks = seq(1,10))
  

fig_print_size = fig_format[[fig_type]][["fig_print_size"]]
fig_width = fig_print_size[1]
fig_height = fig_print_size[2]

fig_filename = 'number_of_request'
ggsave(paste(fig_folder, paste0(fig_filename, '.', plot_type), sep='/'), req_plot,
       device=plot_type, dpi=600, width=fig_width*0.933, height=2.2*fig_size_factor)



# Request Plot
line_color <- c('Hot Water Supply'  = '#3288bd',
                'Hot Water Return'  = '#66c2a5',
                'Boiler Supply Setpoint' = '#f46d43',
                'New G36 Controller Setpoint' = '#fdae61')

line_width <- c('Hot Water Supply'  = .75*fig_size_factor,
                'Hot Water Return'  = .5*fig_size_factor,
                'Boiler Supply Setpoint' = 1.25*fig_size_factor,
                'New G36 Controller Setpoint' = 2*fig_size_factor)

line_type <- c('Hot Water Supply'  = 'solid',
                'Hot Water Return'  = 'solid',
                'Boiler Supply Setpoint' = 'solid',
                'New G36 Controller Setpoint' = 'dashed')

temp_plot <- bind_rows(boiler_temps_bacnet, boiler_setpoint_cdl_calc) %>%
  filter(stream_names != 'New Controller Setpoint [K]',
         Timestamp >= plt_dates[1],
         Timestamp <= plt_dates[2]) %>%
  ggplot(., aes(Timestamp, temps, colour = stream_names, linetype=stream_names, linewidth=stream_names)) +
  fig_format[[fig_type]][["fig_format"]] +
  geom_step() +
  theme(axis.title.x=element_blank(),
        axis.text.x = element_blank()) +
  scale_y_continuous(
    name = "Temperature [°F]",
    breaks = seq(70, 150, 20),
    sec.axis = sec_axis(~ IPtoSI_temp_conversion(.), name='[°C]')
  ) +
  scale_x_datetime(date_breaks = "1 day", labels=date_format("%b %d")) +
  scale_colour_manual(values=line_color, limits=names(line_color)) +
  scale_linewidth_manual(values=line_width, limits=names(line_width)) +
  scale_linetype_manual(values=line_type, limits=names(line_type)) +
  theme(legend.title=element_blank())


fig_print_size = fig_format[[fig_type]][["fig_print_size"]]
fig_width = fig_print_size[1]
fig_height = fig_print_size[2]

fig_filename = 'new_control_setpoint'
ggsave(paste(fig_folder, paste0(fig_filename, '.', plot_type), sep='/'), temp_plot,
       device=plot_type, dpi=600, width=fig_width, height=3.3*fig_size_factor)

#d53e4f

#Combine plots with cowplot
library(cowplot)
combine_plot <- plot_grid(temp_plot, req_plot, align='v', ncol=1, rel_heights = c(1.6,1.07))

fig_filename = 'boiler_temps_combined'
ggsave(paste(fig_folder, paste0(fig_filename, '.', plot_type), sep='/'), combine_plot,
       device=plot_type, dpi=600, width=7, height=2)



df <- bind_rows(num_requests, boiler_setpoint_cdl_calc)


ggplot(df, aes(x=timestamp, value, colour=metric)) +
  geom_step(size=1.25) +
  theme_bw() +
  scale_y_continuous(
    name = "Boiler Setpoint [F]",
    breaks=seq(0,200,20),
    sec.axis= sec_axis(~./coeff, name="Number of Requests")
  ) +
  scale_x_datetime(date_breaks = "12 hour", labels=date_format("%b %d\n%H")) +
  theme(legend.position="top")
