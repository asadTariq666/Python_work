SELECT
	COALESCE(
		CASE
			WHEN s.dataset = 'analytics_abc' AND s.event_date::DATE >= '2023-03-27' THEN
				CASE
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) ILIKE '%es%' THEN 'floyt.com/es'
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) ILIKE '%it%' THEN 'floyt.com/it'
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) ILIKE '%nl%' THEN 'floyt.com/nl'
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) LIKE 'es' THEN 'floyt.com/es' -- to cover http://floyt.com/es
					-- to cover http://floyt.com/es-es or floyt.com/es-it etc
					WHEN lower(split_part(split_part(split_part(landing_page, '//', 2), '/', 2), '-', 2)) LIKE 'es' THEN 'floyt.com/es' 
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) LIKE 'it' THEN 'floyt.com/it'  -- to cover floyt.com/it
					-- to cover http://floyt.com/it-it, it-es, etc
					WHEN lower(split_part(split_part(split_part(landing_page, '//', 2), '/', 2), '-', 2)) LIKE 'it' THEN 'floyt.com/it' 
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) LIKE 'nl' THEN 'floyt.com/nl' -- to cover floyt.com/nl
					-- to cover floyt.com/nl-nl, nl-es, etc
					WHEN lower(split_part(split_part(split_part(landing_page, '//', 2), '/', 2), '-', 2)) LIKE 'nl' THEN 'floyt.com/nl' 
					ELSE NULL
				END
			ELSE 'floyt.com/es'
		END,
		am.app_market,
		NULL
	) AS market,
	COALESCE(
		CASE
			WHEN s.dataset = 'analytics_abc' AND s.event_date::DATE >= '2023-03-27' THEN
				CASE
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) ILIKE '%es%' THEN 'es'
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) ILIKE '%it%' THEN 'it'
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) ILIKE '%nl%' THEN 'nl'
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) LIKE 'es' THEN 'es' -- to cover http://floyt.com/es
					-- to cover http://floyt.com/es-es or floyt.com/es-it etc
					WHEN lower(split_part(split_part(split_part(landing_page, '//', 2), '/', 2), '-', 1)) LIKE 'es' THEN 'es' 
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) LIKE 'it' THEN 'it' -- to cover floyt.com/it
					-- to cover http://floyt.com/it-it, it-es, etc
					WHEN lower(split_part(split_part(split_part(landing_page, '//', 2), '/', 2), '-', 1)) LIKE 'it' THEN 'it' 
					WHEN lower(split_part(split_part(landing_page, '//', 2), '/', 2)) LIKE 'nl' THEN 'nl' -- to cover floyt.com/nl
					 -- to cover floyt.com/nl-nl, nl-es, etc
					WHEN lower(split_part(split_part(split_part(landing_page, '//', 2), '/', 2), '-', 1)) LIKE 'nl' THEN 'nl'
					ELSE NULL
				END
			ELSE 'es'
		END,
		ap.app_language,
		NULL
	) AS "language"
