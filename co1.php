<?php

    $dsn = 'mysql:host=localhost;dbname=graduation';
    $dataBaseUserName = 'root';
    $dataBasePassword = '';
    $option = array(
        PDO::MYSQL_ATTR_INIT_COMMAND => 'SET NAMES utf8'
    );

    try{
        $conn = new PDO($dsn, $dataBaseUserName, $dataBasePassword, $option);
        $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
       echo "Connected successfully";

    }catch (PDOException $e){
       echo "Connection failed: " . $e->getMessage();
    }

?>
 