<?php
 
//login.php

session_start();
 
/**
 * Include ircmaxell's password_compat library.

Include 'lib/password.php';*/
Include 'co1.php';

 if(isset($_POST['submit'])){
    unset($_SESSION['errors']);
    unset($_SESSION['data']);

   $accesscode = filter_var($_POST['accesscode'], FILTER_SANITIZE_STRING);
  //echo $accesscode;
      $error_msg = [];
      if(empty($accesscode)) {
        $error_msg[] = 'access code  is required';
       } else if(strlen($accesscode) !=8) {
       $error_msg[] = 'the access code that entered was not 8 digits long';
     }
     else {
    /* Success */
     }
   
     $data = [
      'accesscode' => $accesscode
    ];
     // if(empty($error_msg)){
         $sql="SELECT * FROM `accesscode` WHERE accesscode=:accesscode";
         $stmt=$conn->prepare($sql);
        $stmt->execute(array('accesscode'=>$accesscode));
        $count=$stmt->rowcount();
        if ($count>0){
           // $_SESSION["accesscode"]=$accesscode;
            header('Location: registration-aaa.php');
              exit();
        }
        else {
          $error_msg[] = 'the access code that entered was not correct';
           $_SESSION['errors'] = $error_msg;
           $_SESSION['data'] = $data;
            header('Location: access-code.php');  
           echo $accesscode;

              exit();
        }
      
}

//}

 
 

