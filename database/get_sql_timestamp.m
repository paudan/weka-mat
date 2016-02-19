function timestamp_obj = get_sql_timestamp(date_obj)
timestamp_obj = javaObject('java.sql.Timestamp',date_obj);
end