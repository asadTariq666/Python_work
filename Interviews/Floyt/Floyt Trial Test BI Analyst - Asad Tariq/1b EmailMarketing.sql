-- Creating new Table

CREATE TABLE IF NOT EXISTS probetag.campaigns_trial_day (
    Campaign_ID string,
    Date DATE,
    Number_of_Bookings INT,
    Number_of_Emails_Send INT,
    Number_of_Emails_Open INT,
    Number_of_Emails_Click INT,
    Number_of_Sessions INT,
    Number_of_Transactions INT,
    Number_of_Bookings_with_Transactions INT,
    Total_Transaction_Values Float64,
    Net_Revenue Float64
);

-- Inserting values into the new table
INSERT INTO probetag.campaigns_trial_day (
    Campaign_ID,
    Date,
    Number_of_Bookings,
    Number_of_Emails_Send,
    Number_of_Emails_Open,
    Number_of_Emails_Click,
    Number_of_Sessions,
    Number_of_Transactions,
    Number_of_Bookings_with_Transactions,
    Total_Transaction_Values,
    Net_Revenue
)
SELECT
    e.email_id AS Campaign_ID,
    CAST(MAX(e.email_created) AS DATE) AS Date,
    COUNT(DISTINCT b.booking_id) AS Number_of_Bookings,
    COUNT(e.customer_id) AS Number_of_Emails_Send,
    COUNT(DISTINCT e.email_open) AS Number_of_Emails_Open,
    COUNT(DISTINCT e.email_click) AS Number_of_Emails_Click,
    SUM(ga.ga_sessions) AS Number_of_Sessions,
    SUM(ga.ga_transactions) AS Number_of_Transactions,
    SUM(CASE WHEN ga.ga_transactions > 0 THEN ga.ga_transactions END) AS Number_of_Bookings_with_Transactions,
    SUM(ga.ga_transactions_revenue) AS Total_Transaction_Values,
    SUM(CASE WHEN ga.ga_transactions > 0 THEN ga.ga_transactions_revenue END) AS Net_Revenue
FROM probetag.emails_trial_day AS e
INNER JOIN probetag.bookings_trial_day b
    ON b.customer_id = e.customer_id
    AND b.booking_is_cancelled = FALSE
INNER JOIN probetag.google_analytics_trial_day AS ga
    ON e.email_id = ga.ga_campaigns
WHERE (e.email_open IS NOT NULL OR e.email_click IS NOT NULL)
    AND (DATE_DIFF(DATE(e.email_open), DATE(b.booking_creation), DAY) + 1 <= 7
    OR DATE_DIFF(DATE(e.email_click), DATE(b.booking_creation), DAY) + 1 <= 7)
GROUP BY e.email_id;