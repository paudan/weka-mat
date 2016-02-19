function conn = connect_db(datasourcename, server, port, dbname, username, password)
% Create a connection to PostgreSQL database
    % Change this string to add more recent version of PostgreSQL JDBC driver
    javaaddpath(sprintf('%slib%spostgresql-9.3-1100.jdbc4.jar',script_dir, filesep));
    logintimeout(5);
    disp('Connecting to database...')
    driver ='org.postgresql.Driver'; %Driver for postgres in MATLAB
    url = ['jdbc:postgresql://' server ':' port '/' dbname]; 
    conn = database(datasourcename, username, password, driver, url);
    ping(conn);
end