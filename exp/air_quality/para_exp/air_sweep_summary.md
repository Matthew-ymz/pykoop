# Air Sweep Summary

????: 80

## Best Run

- run_name: air_all_delay3_lag48
- scope: all
- delay_days: 3
- lag_steps: 48
- station_count: 127
- feature_dim: 1016
- CE: 6.322878
- G_alpha_K: -563.964665
- EC: 8.948052
- selected_r: 2
- runtime_seconds: 40.56

## Top 10 by CE

```text
run_name                     | scope    | province | city | delay_days | lag_steps | station_count | feature_dim | CE       | G_alpha_K   | EC       | selected_r | runtime_seconds
-----------------------------|----------|----------|------|------------|-----------|---------------|-------------|----------|-------------|----------|------------|----------------
air_all_delay3_lag48         | all      |          |      | 3          | 48        | 127           | 1016        | 6.322878 | -563.964665 | 8.948052 | 2          | 40.556702      
air_province_上海_delay5_lag48 | province | 上海市      |      | 5          | 48        | 19            | 228         | 4.353309 | -106.059195 | 6.21504  | 2          | 2.705905       
air_province_浙江_delay5_lag48 | province | 浙江省      |      | 5          | 48        | 44            | 528         | 4.294529 | -229.723309 | 7.415771 | 2          | 10.197823      
air_province_江苏_delay5_lag48 | province | 江苏省      |      | 5          | 48        | 64            | 768         | 4.271844 | -321.954253 | 7.955772 | 2          | 23.198733      
air_all_delay3_lag24         | all      |          |      | 3          | 24        | 127           | 1016        | 3.205762 | -321.258307 | 7.92699  | 2          | 43.469723      
air_city_舟山_delay10_lag48    | city     |          |      | 10         | 48        | 4             | 88          | 2.493162 | -23.678548  | 3.944132 | 2          | 0.965813       
air_city_嘉兴_delay10_lag48    | city     |          |      | 10         | 48        | 4             | 88          | 2.476679 | -22.228036  | 3.890009 | 2          | 0.877403       
air_city_台州_delay10_lag48    | city     |          |      | 10         | 48        | 4             | 88          | 2.476323 | -22.196731  | 3.956575 | 2          | 1.028432       
air_city_湖州_delay10_lag48    | city     |          |      | 10         | 48        | 4             | 88          | 2.467566 | -21.42608   | 3.944485 | 2          | 0.958521       
air_city_镇江_delay10_lag48    | city     |          |      | 10         | 48        | 5             | 110         | 2.459265 | -28.747734  | 4.281844 | 2          | 1.26055        
```

## Average by Scope

```text
scope    | runs | avg_CE   | avg_EC   | avg_G_alpha_K | avg_runtime
---------|------|----------|----------|---------------|------------
all      | 4    | 3.5349   | 9.123611 | -565.778629   | 44.529473  
city     | 64   | 1.767336 | 5.669069 | -75.88894     | 2.156648   
province | 12   | 2.643463 | 7.662397 | -260.544756   | 12.237661  
```