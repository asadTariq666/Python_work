
select * from `aws-silvertours-link`.probetag.bookings_trial_day;

select count(DISTINCT booking_id) as total_departures
from `aws-silvertours-link`.probetag.bookings_trial_day
WHERE booking_is_cancelled is FALSE ;


-- 2.	How many Bookings? 

select count(DISTINCT booking_id) as total_bookings
from `aws-silvertours-link`.probetag.bookings_trial_day;

-- 3.	How many departures? 

select count(DISTINCT booking_id) as total_departures
from `aws-silvertours-link`.probetag.bookings_trial_day
WHERE booking_is_cancelled is FALSE ;

-- 4. How many customers? 
select count(DISTINCT customer_id) as total_customers
from `aws-silvertours-link`.probetag.bookings_trial_day;


-- 5.	For each booking, what is the average price per day based on the trip length? 

SELECT booking_id, 
booking_price / (DATE_DIFF(DATE(booking_dest_date), DATE(booking_dep_date), DAY) + 1) AS average_price_per_day
FROM  `aws-silvertours-link`.probetag.bookings_trial_day;

------------------------------------------------------------
-- With Common table Expression
WITH price_per_booking AS (
  SELECT 
    booking_id, booking_price,
    DATE_DIFF(DATE(booking_dest_date), DATE(booking_dep_date), DAY) + 1 AS booking_duration
  FROM 
    `aws-silvertours-link`.probetag.bookings_trial_day
)
SELECT 
  booking_id, 
  booking_price / booking_duration AS average_price_per_day
FROM 
  price_per_booking;
 
 
-- 6.	For each booking, what is the total transaction value? 
 
SELECT booking_id, 
	   (booking_price + additional_travel_insurance_price + additional_deductible_insurance_price) AS total_transaction_value
FROM `aws-silvertours-link`.probetag.bookings_trial_day;

-- 7.	For each booking, what is the net revenue? 

 SELECT booking_id, 
	   (booking_supplier_commission  + additional_travel_insurance_commission  + additional_deductible_insurance_commission) 
	   AS total_net_revenue
FROM `aws-silvertours-link`.probetag.bookings_trial_day;
 





