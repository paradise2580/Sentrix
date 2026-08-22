#!/bin/bash
# Restarts MariaDB if it isn't already running. Needed because this sandbox
# doesn't persist background processes across separate tool invocations —
# NOT something a real server/deployment needs to worry about.
if ! mysqladmin ping -h 127.0.0.1 --silent 2>/dev/null; then
    rm -f /run/mysqld/mysqld.pid /run/mysqld/mysqld.sock 2>/dev/null
    mkdir -p /run/mysqld && chown mysql:mysql /run/mysqld
    su mysql -s /bin/bash -c "/usr/sbin/mariadbd --datadir=/var/lib/mysql --socket=/run/mysqld/mysqld.sock" > /tmp/mariadb_start.log 2>&1 &
    for i in $(seq 1 10); do
        sleep 1
        mysqladmin ping -h 127.0.0.1 --silent 2>/dev/null && break
    done
fi
mysqladmin ping -h 127.0.0.1 --silent && echo "MariaDB is up"
