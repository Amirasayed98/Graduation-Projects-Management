<?php
 
//login.php

session_start();
 
/**
 * Includ e ircmaxell's password_compat library.

Include 'lib/password.php';*/
Include 'co1.php';

 if(isset($_POST['submit'])){
    unset($_SESSION['errors']);
    unset($_SESSION['data']);
   $accesscode = filter_var($_POST['accesscode'], FILTER_SANITIZE_STRING);
   $email = filter_var($_POST['email'], FILTER_SANITIZE_STRING);
   $password= filter_var($_POST['password'], FILTER_SANITIZE_STRING);
  //echo $password;
      $error_msg = [];
      if(empty($email )){
        $error_msg[] = 'email address  and password  is required';
      }
      if (strlen($password) <= '8') {
            $error_msg[] = "Your Password Must Contain At Least 8 Characters!";
        }
        elseif(!preg_match("#[0-9]+#",$password)) {
            $error_msg[] = "Your Password Must Contain At Least 1 Number!";
        }
        elseif(!preg_match("#[A-Z]+#",$password)) {
            $error_msg[] = "Your Password Must Contain At Least 1 Capital Letter!";
        }
        elseif(!preg_match("#[a-z]+#",$password)) {
            $error_msg[] = "Your Password Must Contain At Least 1 Lowercase Letter!";
        } 
        elseif(!preg_match('/[\'^£$%&*()}{@#~?><>,|=_+¬-]/', $password)) {
        $error_msg[]= "Your Password Must Contain At Least 1 Special Character !"."<br>";
         }
        else {
           
        }
  
     $data = [
      'accesscode'=>$accesscode,
      'email' => $email
      ,'password'=>$password
    ];
      //if(empty($error_msg)){
         $sql="SELECT  accesscode,email ,password  FROM `company` WHERE accesscode=:accesscode and email=:email and password=:password";
         $stmt=$conn->prepare($sql);
        $stmt->execute(array('accesscode'=>$accesscode,'email'=>$email,'password'=>$password ));
        $count=$stmt->rowcount();
         
        if ($count>0){
            $sql2="SELECT  *  FROM `company` WHERE accesscode=:accesscode and email=:email and password=:password";
            $stmt2=$conn->prepare($sql2);
           $stmt2->execute(array('accesscode'=>$accesscode,'email'=>$email,'password'=>$password ));
          $r=$stmt2->fetch(PDO::FETCH_ASSOC);
          $cid=$r['company_id'];
          $da=['cid'=>$cid];
            $_SESSION['da'] = $da;
            header('Location:ai-cards-explore.php');
              exit();
        }
        else {
          $error_msg[] = 'the email or password that entered was not correct';
           $_SESSION['errors'] = $error_msg;
           $_SESSION['data'] = $data;
            header('Location: login.php');  
           echo $accesscode;

              exit();
        }
      
//}

}

 
?>

