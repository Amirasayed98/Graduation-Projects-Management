<?php

session_start();
Include 'co1.php';

 if(isset($_POST['submit'])){
    unset($_SESSION['errors']);
    unset($_SESSION['data']);

   $phone = filter_var($_POST['phone'], FILTER_SANITIZE_STRING);
   echo $phone;
  //echo $accesscode;
      $error_msg = [];
       if($phone!=''){
          if(strlen($phone) != 11 && !preg_match('/01[0125]{1}[0-9]{8}/', $phone ))
            $error_msg[] = ' invalid phone number';
       }
   
     $data = [
      'phone' => $phone
    ];
     // if(empty($error_msg)){
      $sql="SELECT * FROM `company` WHERE phone=:phone";
      $stmt=$conn->prepare($sql);
        $stmt->execute(array('phone'=>$phone));
        $count=$stmt->rowcount();
        if ($count>0){
           // $_SESSION["accesscode"]=$accesscode;
          $da=['phone'=>$phone];
          $_SESSION['da'] = $da;
            header('Location: update-password.php');
              exit();
        }
        else {
          $error_msg[] = 'the phone that entered was not correct';
           $_SESSION['errors'] = $error_msg;
           $_SESSION['data'] = $data;
            header('Location: forgot-password.php');
         
              exit();
        }
      
}

//}
?>