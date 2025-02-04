SELECT *, 
       DATEFROMPARTS(year, month, day) AS TrxDate
FROM [Yaumi_Forecast_RouteItem_2025-01-10_2Routes_updated_3003_3004]
WHERE CustomerCode IN ('3003', '3004') 
AND ItemCode IN ('50-4072', '50-0261', '50-4085', '50-0276', '50-0102', 
                 '50-0117', '50-0526', '50-0401', '50-0412', '50-0296')
ORDER BY TrxDate;