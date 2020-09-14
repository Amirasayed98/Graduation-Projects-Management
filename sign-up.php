<?php
    session_start();
?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>sign up</title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    <link rel="stylesheet" href="css/sign up.css"/>
    
    <script src="js/jquery-3.4.1.min.js"></script>
    <script src="js/bootstrap.min.js"></script>
    
    
</head>
<body>
   <span id="error">

                     <?php
                     if(isset($_SESSION['errors'])){
                         if(!empty($_SESSION['errors'])){
                             echo '<div class="alert alert-danger">';
                             echo '<ul>';
                             foreach ($_SESSION['errors'] as $error){
                                 echo '<li>' ;
                                    echo $error;
                                 echo '</li>' ;
                             }
                             echo '</ul>';
                             echo   '</div>';
        }
        }
        ?>
        </span>
 
    <form action="sign-up-data.php" method="post" enctype="multipart/form-data">
        
      <img src="image/logo-main.PNG" class="card-img-top" alt="find image">
        <div class="container">
           <div class="row">
              <div class="col">
                <input type="text" placeholder="Company  name*" name="companyname" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['companyname'] ?>">
              </div>
           </div>
           <div class="row">
              <div class="col">
                <input type="text" placeholder="Email* (example:bfcai@gmail.com)" name="email" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['email'] ?>">  
              </div>
           </div>
           <div class="row">
              <div class="col">
                <input type="text" placeholder="Phone*  (must be 11 numbers like: 01045626578)" name="phone" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['phone'] ?>">
              </div>
           </div>

         <div class="form-row" >
            
             <div class="form-group col-md-6"style="margin-bottom: 5px; margin-top:5px;">
              <select class="custom-select" name="city" id="trackOpt" >
                <option selected disabled value="" >City*</option>
                <option value="Alexandria"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Alexandria') echo 'selected'?>>Alexandria</option>
                <option value="Aswan"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Aswan') echo 'selected'?>>Aswan</option>
                <option value="Asyut"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Asyut') echo 'selected'?>>Asyut</option>
                <option value="Beheira"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Beheira') echo 'selected'?>>Beheira</option>
                <option value="Beni Suef"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Beni Suef') echo 'selected'?>>Beni Suef</option>
                <option value="Cairo"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Cairo') echo 'selected'?>>Cairo</option>
                <option value="Dakahlia"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Dakahlia') echo 'selected'?>>Dakahlia</option>
                <option value="Damietta"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Damietta') echo 'selected'?>>Damietta</option>
                <option value="Fayoum"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Fayoum') echo 'selected'?>>Fayoum</option>
                <option value="Gharbia"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Gharbia') echo 'selected'?>>Gharbia</option>
                <option value="Giza"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Giza') echo 'selected'?>>Giza</option>
                <option value="Ismailia"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Ismailia') echo 'selected'?>>Ismailia</option>
                <option value="Kafr El-Sheikh"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Kafr El-Sheikh') echo 'selected'?>>Kafr El-Sheikh</option>
                <option value="Loxur"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Loxur') echo 'selected'?>>Loxur</option>
                <option value="Matruh"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Matruh') echo 'selected'?>>Matruh</option>
                <option value="Minya"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Minya') echo 'selected'?>>Minya</option>
                <option value="Monufia"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Monufia') echo 'selected'?>>Monufia</option>
                <option value="New Valley"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'New Valley') echo 'selected'?>>New Valley</option>
                <option value="North Sinai"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'North Sinai') echo 'selected'?>>North Sinai</option>
                <option value="Port Said"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Port Said') echo 'selected'?>>Port Said</option>
                <option value="Qalyubia"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Qalyubia') echo 'selected'?>>Qalyubia</option>
                <option value="Qena"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Qena') echo 'selected'?>>Qena</option>
                <option value="Red Sea"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Red Sea') echo 'selected'?>>Red Sea</option>
                <option value="Sharqia"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Sharqia') echo 'selected'?>>Sharqia</option>
                <option value="Sohag"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Sohag') echo 'selected'?>>Sohag</option>
                <option value="South Sinai"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'South Sinai') echo 'selected'?>>South Sinai</option>
                <option value="Suez"<?php if(isset($_SESSION['data']) && $_SESSION['data']['city'] == 'Suez') echo 'selected'?>>Suez</option>
            
          </select>
          </div>
          <div class="form-group col-md-6" style="margin-bottom: 5px; margin-top:5px;">
              <input type="text" placeholder="Area*" name="area" style="margin-top: 0px;margin-bottom: 0px;height: 38px;"  required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['area'] ?>">
            
          </div>
          </div>
          
          <div class="row">
            <div class="col">
              <input type="password" placeholder="Create password*" name="psw1"  required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['psw1'] ?>">
            </div>
          </div>
          <div class="row">
            <div class="col note">*Use 8 or more characters (mix of letters, numbers & symbols) </div>
          </div>
          <div class="row">
            <div class="col">
              <input type="password" placeholder="Confirm password*" name="psw2"  required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['psw2'] ?>">
            </div>
          </div>   
          <div class="sign"> 
             <button type="submit" >Sign Up</button>
             <a href="login.php">Already have an account</a>
             <a href="Home.php">Home</a>
          
        </div>
    </div>
  </div>    
        
    </form>

     

</body>
</html>
<?php
if(isset($_SESSION['errors'])){
    unset($_SESSION['errors']);
}
if(isset($_SESSION['data'])){
    unset($_SESSION['data']);
}
?>

