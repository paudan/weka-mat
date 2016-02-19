function [dataset, attributes]=query_data(querystr)
    conn = connect_db;
    setdbprefs('DataReturnFormat','cellarray');
    curs = exec(conn, querystr);
    curs = fetch(curs);
    dataset=curs.Data;
    attributes = attr(curs);
    close(curs);
    close(conn);
end